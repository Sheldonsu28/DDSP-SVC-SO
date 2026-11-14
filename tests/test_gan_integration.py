"""A bounded real SoftVcStylizer/full-discriminator/DDSP training and resume check."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import soundfile as sf
import torch

from ddsp.checkpoint import DDSPOnlyUnit2Wav
from ddsp.vocoder import CombSubSuperFast
from discriminators.discriminator import Discriminator
from frontend import load_frontend
from logger import utils
from nsf_hifigan.nvSTFT import STFT
from optimizer.muon import Muon_AdamW
from reflow.solver import train_vc_osgan
from stylizer.stylizer import SoftVcStylizer


class TestVocoder:
    def __init__(self):
        self.stft = STFT(44100, 128, 2048, 2048, 512, 40, 22050)

    def extract(self, waveform):
        return self.stft.get_mel(waveform).transpose(1, 2)

    def infer(self, mel, f0):
        return mel.mean(-1).repeat_interleave(512, dim=1).unsqueeze(1)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA integration requires a GPU")
class GANIntegrationTests(unittest.TestCase):
    def test_train_validate_save_and_resume_frozen_ddsp_with_full_discriminator(self):
        self.run_training("fp32")

    def test_bfloat16_training_validation_and_resume(self):
        self.run_training("bf16")

    def test_float16_training_validation_and_resume(self):
        self.run_training("fp16")

    def run_training(self, precision):
        torch.manual_seed(12)
        torch.set_num_threads(2)
        steps = 12 if precision == "fp16" else 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "val/audio").mkdir(parents=True)
            waveform = (0.1 * torch.sin(torch.arange(4096) * 2 * torch.pi * 220 / 44100)).unsqueeze(0).cuda()
            sf.write(root / "val/audio/sample.wav", waveform[0].cpu().numpy(), 44100)
            model_settings = dict(latent_dim=16, decoder_max_loops=1)
            style = SoftVcStylizer(**model_settings).cuda()
            discriminator = Discriminator({}).cuda()
            base = DDSPOnlyUnit2Wav(CombSubSuperFast(44100, 512, 2048, n_unit=768,
                n_spk=1, num_layers=1, dim_model=16, use_pitch_aug=True)).cuda()
            vocoder = TestVocoder()
            data = dict(units=torch.randn(1, 8, 768, device="cuda"),
                units_ori=torch.randn(1, 8, 768, device="cuda"),
                units_w=torch.randn(1, 8, 1280, device="cuda"),
                units_h=torch.randn(1, 8, 256, device="cuda"),
                spk_embd=torch.randn(1, 192, device="cuda"),
                mel=vocoder.extract(waveform), gt_audio=waveform,
                f0=torch.full((1, 8, 1), 220., device="cuda"),
                volume=torch.full((1, 8, 1), .1, device="cuda"),
                spk_id=torch.ones((1, 1), dtype=torch.long, device="cuda"),
                aug_shift=torch.zeros((1, 1, 1), device="cuda"),
                name=["sample"], name_ext=["sample.wav"])
            args = utils.DotDict(dict(device="cuda", env=dict(expdir=str(root / "base"), style_dir=str(root / "frontend")),
                data=dict(sampling_rate=44100, block_size=512, valid_path=str(root / "val")),
                model=dict(t_start=1.0), frontend=dict(model=model_settings),
                train=dict(epochs=1, amp_dtype=precision, lambda_ddsp=1., interval_log=1,
                    interval_val=steps, interval_force_save=steps * 10, save_opt=True),
                infer=dict(infer_step=2, method="euler")))
            optimizer = Muon_AdamW(style, muon_args={"weight_decay": 0.1}, adamw_args={"weight_decay": 0})
            optimizer_d = Muon_AdamW(discriminator, muon_args={"weight_decay": 0.1}, adamw_args={"weight_decay": 0})
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 100)
            scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, 100)
            gradients = []
            handle = style.output_proj.weight_v.register_hook(lambda value: gradients.append(value.detach().clone()))
            initial_weight = style.output_proj.weight_v.detach().clone()
            batches = [{key: value.clone() if torch.is_tensor(value) else value for key, value in data.items()}
                       for _ in range(steps)]
            train_vc_osgan(args, 0, style, base, discriminator, optimizer, optimizer_d,
                           scheduler, scheduler_d, vocoder, batches, [data])
            handle.remove()
            self.assertTrue(gradients)
            finite_gradients = [gradient for gradient in gradients if torch.isfinite(gradient).all()]
            self.assertTrue(finite_gradients, "FP16 scale calibration never recovered finite gradients")
            self.assertGreater(finite_gradients[-1].abs().sum().item(), 0)
            self.assertFalse(torch.equal(initial_weight, style.output_proj.weight_v))
            self.assertTrue(all(torch.isfinite(parameter).all() for parameter in style.parameters()))
            self.assertTrue(all(torch.isfinite(parameter).all() for parameter in discriminator.parameters()))
            self.assertTrue(all(parameter.grad is None for parameter in base.parameters()))
            checkpoint = root / f"frontend/model_{steps}.pt"
            self.assertTrue(checkpoint.is_file())
            restored = load_frontend(checkpoint, "cpu")
            self.assertEqual(restored.latent_dim, 16)
            step, *_ = utils.load_gan_model(checkpoint, style, optimizer,
                                             discriminator, optimizer_d, device="cuda")
            self.assertEqual(step, steps)
            self.assertTrue(optimizer.state_dict())
            self.assertTrue(optimizer_d.state_dict())
            with (root / "frontend/log_info.txt").open() as file:
                self.assertIn("validation", file.read())
        del style, discriminator, base, optimizer, optimizer_d
        torch.cuda.empty_cache()


if __name__ == "__main__":
    unittest.main()
