import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from ddsp.checkpoint import DDSPOnlyUnit2Wav, load_ddsp_model
from ddsp.vocoder import CombSubSuperFast


class LoadDDSPModelTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.model_path = self.directory / 'model.pt'
        self.config_path = self.directory / 'config.yaml'
        self.config = {
            'data': {
                'sampling_rate': 16000,
                'block_size': 16,
                'encoder_out_channels': 8,
            },
            'model': {
                'type': 'RectifiedFlow',
                'win_length': 64,
                'n_spk': 2,
                'n_aux_layers': 1,
                'n_aux_chans': 16,
                'use_norm': False,
                'use_attention': False,
                'use_pitch_aug': True,
            },
        }
        self.config_path.write_text(yaml.safe_dump(self.config), encoding='utf-8')
        self.source = CombSubSuperFast(
            16000, 16, 64, n_unit=8, n_spk=2, num_layers=1,
            dim_model=16, use_pitch_aug=True,
        ).eval()
        self.state_dict = {
            'ddsp_model.' + key: value
            for key, value in self.source.state_dict().items()
        }
        # An unrelated reflow architecture must not affect DDSP loading.
        self.state_dict['reflow_model.unused.weight'] = torch.ones(3)
        self.save_checkpoint()

    def save_checkpoint(self):
        torch.save({'model': self.state_dict, 'global_step': 10}, self.model_path)

    def test_loads_only_ddsp_and_matches_source_audio(self):
        model, args = load_ddsp_model(
            self.model_path, device=torch.device('cpu'), wrapped=False,
        )
        self.assertIsInstance(model, CombSubSuperFast)
        self.assertFalse(model.training)
        self.assertFalse(hasattr(model, 'reflow_model'))
        self.assertEqual(args.data.sampling_rate, 16000)
        for key, value in self.source.state_dict().items():
            torch.testing.assert_close(model.state_dict()[key], value)

        units = torch.randn(1, 8, 8)
        f0 = torch.full((1, 8, 1), 220.0)
        volume = torch.full((1, 8, 1), 0.1)
        spk_id = torch.tensor([[2]])
        with torch.inference_mode():
            torch.manual_seed(123)
            expected, _ = self.source(units, f0, volume, spk_id=spk_id)
            torch.manual_seed(123)
            actual, _ = model(units, f0, volume, spk_id=spk_id)
        self.assertEqual(actual.shape, (1, 128))
        torch.testing.assert_close(actual, expected)

    def test_explicit_config_path(self):
        custom_path = self.directory / 'custom.yaml'
        custom_path.write_text(yaml.safe_dump(self.config), encoding='utf-8')
        self.config_path.write_text('model: {type: WrongModel}', encoding='utf-8')
        model, _ = load_ddsp_model(str(self.model_path), config_path=str(custom_path))
        self.assertFalse(model.training)

    def test_explicit_config_directory(self):
        config_dir = self.directory / 'custom_config'
        config_dir.mkdir()
        (config_dir / 'config.yaml').write_text(
            yaml.safe_dump(self.config), encoding='utf-8',
        )
        self.config_path.write_text('model: {type: WrongModel}', encoding='utf-8')
        model, args = load_ddsp_model(self.model_path, config_path=config_dir)
        self.assertIsInstance(model, DDSPOnlyUnit2Wav)
        self.assertEqual(args.data.sampling_rate, 16000)

    def test_checkpoint_folder_can_also_be_config_directory(self):
        torch.save(
            {'model': self.state_dict, 'global_step': 10},
            self.directory / 'model_10.pt',
        )
        model, args = load_ddsp_model(
            str(self.directory), device='cpu', config_path=str(self.directory),
        )
        self.assertIsInstance(model, DDSPOnlyUnit2Wav)
        self.assertEqual(args.data.sampling_rate, 16000)

    def test_folder_loads_largest_numeric_step_with_local_config(self):
        key = 'ddsp_model.unit2ctrl.volume_embed.bias'
        for step in (1000, 90, 9):
            state_dict = dict(self.state_dict)
            state_dict[key] = torch.full_like(state_dict[key], step)
            torch.save(
                {'model': state_dict, 'global_step': step},
                self.directory / f'model_{step}.pt',
            )
        # Ignore other checkpoint names, malformed steps, and directories.
        for name in ('model_best.pt', 'other_99999.pt', 'model_99999_.pt'):
            (self.directory / name).write_bytes(b'not a checkpoint')
        (self.directory / 'model_99999.pt').mkdir()

        model, args = load_ddsp_model(str(self.directory))
        self.assertIsInstance(model, DDSPOnlyUnit2Wav)
        self.assertFalse(model.training)
        self.assertEqual(args.data.sampling_rate, self.config['data']['sampling_rate'])
        torch.testing.assert_close(
            model.ddsp_model.unit2ctrl.volume_embed.bias,
            torch.full_like(self.state_dict[key], 1000),
        )

    def test_folder_without_numbered_checkpoints_reports_error(self):
        # setUp creates model.pt, which has no numeric step and is ignored.
        (self.directory / 'model_best.pt').write_bytes(b'not a checkpoint')
        with self.assertRaisesRegex(FileNotFoundError, r'No model_<step>\.pt checkpoints'):
            load_ddsp_model(self.directory)

    def test_rejects_checkpoint_without_ddsp_weights(self):
        torch.save({'model': {'reflow_model.weight': torch.ones(1)}}, self.model_path)
        with self.assertRaisesRegex(ValueError, r'No ddsp_model\.\* weights'):
            load_ddsp_model(self.model_path)

    def test_rejects_incomplete_ddsp_weights(self):
        del self.state_dict['ddsp_model.unit2ctrl.volume_embed.weight']
        self.save_checkpoint()
        with self.assertRaisesRegex(RuntimeError, 'Missing key'):
            load_ddsp_model(self.model_path)

    def test_wrapper_training_supports_backward_and_optimizer_step(self):
        model, _ = load_ddsp_model(self.model_path)
        self.assertIsInstance(model, DDSPOnlyUnit2Wav)
        self.assertFalse(model.training)
        self.assertFalse(hasattr(model, 'reflow_model'))
        self.assertEqual(set(model.state_dict()), set(self.state_dict) - {
            'reflow_model.unused.weight',
        })
        model.train()
        self.assertTrue(model.ddsp_model.training)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        units = torch.randn(1, 8, 8, requires_grad=True)
        vocoder = FrameVocoder(block_size=16)
        target = torch.ones(1, 8, 1)
        optimizer.zero_grad()
        loss, reflow_loss, band_loss, waveform, mel = model(
            units, torch.full((1, 8, 1), 220.0), torch.full((1, 8, 1), 0.1),
            torch.tensor([[2]]), spk_mix_dict={1: 0.25, 2: 0.75},
            aug_shift=torch.tensor([[1.0]]), vocoder=vocoder, gt_spec=target,
            infer=False, return_wav=True, infer_step=50, method='rk4',
            t_start=0.25, silence_front=0, use_tqdm=True,
        )
        self.assertEqual(waveform.shape, (1, 128))
        self.assertEqual(mel.shape, target.shape)
        torch.testing.assert_close(loss, torch.nn.functional.mse_loss(mel, target))
        for placeholder in (reflow_loss, band_loss):
            self.assertEqual(placeholder.item(), 0.0)
            self.assertEqual(placeholder.device, loss.device)
            self.assertEqual(placeholder.dtype, loss.dtype)
        (loss + reflow_loss + band_loss).backward()
        self.assertIsNotNone(units.grad)
        self.assertTrue(torch.isfinite(units.grad).all())
        self.assertGreater(units.grad.abs().sum().item(), 0.0)
        parameter = model.ddsp_model.unit2ctrl.volume_embed.weight
        self.assertIsNotNone(parameter.grad)
        self.assertTrue(torch.isfinite(parameter.grad).all())
        before = parameter.detach().clone()
        optimizer.step()
        self.assertGreater((parameter.detach() - before).abs().sum().item(), 0.0)

    def test_wrapper_inference_matches_ddsp_mel_and_vocoder_audio(self):
        model, _ = load_ddsp_model(self.model_path)
        units = torch.randn(1, 8, 8)
        f0 = torch.full((1, 8, 1), 220.0)
        volume = torch.full((1, 8, 1), 0.1)
        spk_id = torch.tensor([[1]])
        vocoder = FrameVocoder(block_size=16)
        silence_front = 2 * 16 / 16000
        with torch.inference_mode():
            torch.manual_seed(123)
            waveform, _ = model.ddsp_model(units, f0, volume, spk_id)
            expected = vocoder.extract(waveform[:, 32:])
            torch.manual_seed(123)
            actual = model(
                units, f0, volume, spk_id, vocoder=vocoder,
                silence_front=silence_front, infer_step=100, method='unused',
                t_start=0.0, use_tqdm=True,
            )
            torch.testing.assert_close(actual, expected)
            torch.manual_seed(123)
            audio = model(
                units, f0, volume, spk_id, vocoder=vocoder,
                silence_front=silence_front, return_wav=True,
            )
            torch.testing.assert_close(audio, vocoder.infer(expected, f0[:, -6:]))
            torch.manual_seed(123)
            raw = model(
                units, f0, volume, spk_id, silence_front=silence_front,
                return_wav=True,
            )
            torch.testing.assert_close(raw, waveform[:, 32:])

    def test_wrapper_reports_missing_training_inputs(self):
        model, _ = load_ddsp_model(self.model_path)
        for inputs in ({}, {'vocoder': FrameVocoder(16)}, {'gt_spec': torch.zeros(1)}):
            with self.subTest(inputs=list(inputs)):
                with self.assertRaisesRegex(ValueError, 'requires vocoder and gt_spec'):
                    model(None, None, None, infer=False, **inputs)

    def test_validation_accepts_scalar_augmentation(self):
        model, _ = load_ddsp_model(self.model_path)
        model.requires_grad_(False)
        units = torch.randn(2, 8, 8)
        f0 = torch.full((2, 8, 1), 220.0)
        volume = torch.full((2, 8, 1), 0.1)
        spk_id = torch.tensor([[1], [2]])
        vocoder = FrameVocoder(16)
        target = torch.zeros(2, 8, 1)
        with torch.no_grad():
            for shift in (0, 0.75, torch.tensor(0), torch.tensor(1.25, dtype=torch.float64)):
                with self.subTest(shift=shift):
                    shaped = torch.full((2, 8, 1), float(shift))
                    torch.manual_seed(123)
                    expected = model(
                        units, f0, volume, spk_id, aug_shift=shaped, vocoder=vocoder,
                    )
                    torch.manual_seed(123)
                    mel = model(
                        units, f0, volume, spk_id, aug_shift=shift, vocoder=vocoder,
                    )
                    torch.testing.assert_close(mel, expected)
                    torch.manual_seed(123)
                    loss, _, _, waveform, loss_mel = model(
                        units, f0, volume, spk_id, aug_shift=shift, vocoder=vocoder,
                        gt_spec=target, infer=False,
                    )
                    torch.testing.assert_close(loss_mel, expected)
                    torch.testing.assert_close(loss, expected.square().mean())
                    self.assertEqual(waveform.shape, (2, 128))

    def test_augmentation_broadcasts_per_example_without_mixing_batch_and_time(self):
        model, _ = load_ddsp_model(self.model_path)
        units = torch.randn(2, 8, 8)
        f0 = torch.full((2, 8, 1), 220.0)
        volume = torch.full((2, 8, 1), 0.1)
        spk_id = torch.tensor([[1], [2]])
        per_example = torch.tensor([0.5, -1.0])
        per_frame = per_example[:, None, None].expand(-1, 8, -1)
        with torch.inference_mode():
            torch.manual_seed(123)
            expected = model(
                units, f0, volume, spk_id, aug_shift=per_frame, return_wav=True,
            )
            for shift in (per_example, per_example[:, None],
                          per_example[:, None, None], per_frame[..., 0], per_frame):
                with self.subTest(shape=tuple(shift.shape)):
                    torch.manual_seed(123)
                    waveform = model(
                        units, f0, volume, spk_id, aug_shift=shift, return_wav=True,
                    )
                    torch.testing.assert_close(waveform, expected)

    def test_scalar_augmentation_keeps_gradients_through_frozen_ddsp(self):
        model, _ = load_ddsp_model(self.model_path)
        model.requires_grad_(False)
        shift = torch.tensor(1.25, dtype=torch.float64, requires_grad=True)
        units = torch.randn(1, 8, 8, requires_grad=True)
        loss, _, _, _, _ = model(
            units, torch.full((1, 8, 1), 220.0), torch.full((1, 8, 1), 0.1),
            torch.tensor([[1]]), aug_shift=shift, vocoder=FrameVocoder(16),
            gt_spec=torch.ones(1, 8, 1), infer=False,
        )
        loss.backward()
        for gradient in (shift.grad, units.grad):
            self.assertIsNotNone(gradient)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.0)


class FrameVocoder:
    """Small differentiable feature extractor for wrapper integration tests."""

    def __init__(self, block_size):
        self.block_size = block_size

    def extract(self, audio):
        return audio.unfold(1, self.block_size, self.block_size).square().mean(-1, keepdim=True)

    def infer(self, mel, f0):
        return mel.squeeze(-1).repeat_interleave(self.block_size, dim=1) + f0.mean() * 0


if __name__ == '__main__':
    unittest.main()
