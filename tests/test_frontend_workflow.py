from pathlib import Path
import pickle
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from frontend import FEATURE_STREAMS, apply_overrides, load_indexes, retrieve
from feature_retrival.utils import train_index
from logger.utils import DotDict
from preprocess import speaker_model_input, save_speaker_means
from reflow.data_loaders import AudioDataset


def fixture_dataset(root, sample_rate=44100, hop=512, frames=110, frontend=True):
    root = Path(root)
    (root / "audio").mkdir(parents=True, exist_ok=True)
    waveform = np.linspace(-0.5, 0.5, frames * hop, dtype=np.float32)
    sf.write(root / "audio/sample.wav", waveform, sample_rate, subtype="FLOAT")
    shapes = dict(units=768, mel=128, aug_mel=128)
    if frontend:
        shapes.update(whisper_units=1280, hubert_units=256)
    for key, channels in shapes.items():
        (root / key).mkdir(exist_ok=True)
        values = np.arange(frames, dtype=np.float16)[:, None].repeat(channels, axis=1)
        np.save(root / key / "sample.wav.npy", values)
    for key, value in (("f0", 220), ("volume", 0.1), ("aug_vol", 0.1)):
        (root / key).mkdir(exist_ok=True)
        np.save(root / key / "sample.wav.npy", np.full(frames, value, dtype=np.float32))
    if frontend:
        (root / "speaker").mkdir(exist_ok=True)
        np.save(root / "speaker/sample.wav.npy", np.ones(192, dtype=np.float32))
        save_speaker_means(root, ["sample.wav"])
    np.save(root / "pitch_aug_dict.npy", {"sample.wav": 0})
    return waveform


class FrontendDataTests(unittest.TestCase):
    def test_cached_and_uncached_audio_crops_follow_feature_timestamps(self):
        for sample_rate, hop in ((44100, 512), (48000, 240)):
            for cache in (False, True):
                with self.subTest(sample_rate=sample_rate, hop=hop, cache=cache), tempfile.TemporaryDirectory() as directory:
                    waveform = fixture_dataset(directory, sample_rate, hop)
                    dataset = AudioDataset(directory, 3.1 * hop / sample_rate, hop, sample_rate,
                        load_all_data=cache, train_frontend=True, load_audio=True, fp16=False)
                    with patch("reflow.data_loaders.random.randint", return_value=100):
                        item = dataset[0]
                    self.assertEqual(item["units_ori"][0, 0].item(), 100)
                    for key in ("units", "units_ori", "units_w", "units_h", "mel"):
                        self.assertEqual(item[key].dtype, torch.float32)
                    torch.testing.assert_close(item["units"], item["units_ori"])
                    torch.testing.assert_close(item["gt_audio"], torch.from_numpy(waveform[100 * hop:103 * hop]))

    def test_upstream_dataset_needs_no_frontend_assets(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture_dataset(directory, frontend=False)
            dataset = AudioDataset(directory, 0.2, 512, 44100, train_frontend=False, fp16=False)
            item = dataset[0]
            self.assertNotIn("spk_embd", item)
            self.assertEqual(item["units"].dtype, torch.float32)

    def test_speaker_resampling_preserves_duration_and_original_waveform(self):
        for rate in (16000, 44100, 48000):
            audio = np.sin(2 * np.pi * 440 * np.arange(rate) / rate).astype(np.float32)
            original = audio.copy()
            converted = speaker_model_input(audio, rate, "cpu")
            self.assertEqual(converted.shape, (1, 16000))
            self.assertEqual(converted.device.type, "cpu")
            self.assertEqual(converted.dtype, torch.float32)
            np.testing.assert_array_equal(audio, original)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_speaker_resampling_respects_cuda(self):
        converted = speaker_model_input(np.ones(44100, dtype=np.float32), 44100, "cuda")
        self.assertEqual(converted.shape, (1, 16000))
        self.assertEqual(converted.device.type, "cuda")

    def test_cli_resources_override_config(self):
        from argparse import Namespace
        args = DotDict({"frontend": {"checkpoint": "old.pt", "index_dir": "old"}})
        settings = apply_overrides(args, Namespace(frontend_checkpoint="new.pt", index_dir="new", speaker_embedding="voice.npy"))
        self.assertEqual(settings.checkpoint, "new.pt")
        self.assertEqual(settings.index_dir, "new")
        self.assertEqual(settings.speaker_embedding, "voice.npy")
        self.assertTrue(settings.enabled)


class IndexTests(unittest.TestCase):
    def test_three_fp16_indexes_reload_and_retrieve_with_original_formula(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(4)
            data = rng.standard_normal((12, 8)).astype(np.float16)
            output = root / "indexes"
            output.mkdir()
            for stream, filename in FEATURE_STREAMS.items():
                for split in ("train", "val"):
                    folder = root / split / stream
                    folder.mkdir(parents=True)
                    np.save(folder / "sample.wav.npy", data)
                index = train_index("", directory, feat=stream)
                self.assertEqual(index.ntotal, 24)
                self.assertEqual(index.d, 8)
                with (output / filename).open("wb") as file:
                    pickle.dump({1: index}, file)
            indexes = load_indexes({"index_dir": str(output)}, 1)
            features = torch.from_numpy(data.astype(np.float32)).unsqueeze(0)
            mapped = retrieve(features, indexes["units"], 4)
            index, vectors = indexes["units"]
            _, neighbors = index.search(features[0].numpy(), 4)
            target = torch.from_numpy(vectors[neighbors].mean(axis=1))
            source_std, source_mean = torch.std_mean(features, dim=1)
            target_std, target_mean = torch.std_mean(target, dim=0)
            expected = (features - source_mean) / source_std * target_std + target_mean
            torch.testing.assert_close(mapped, expected)


class Mod13Tests(unittest.TestCase):
    def test_mod13_forward_backward_and_auxiliary_decoder_losses(self):
        from stylizer.stylizer import SoftVcStylizer
        from reflow.solver import _stylizer_forward_with_decoder_aux, _decoder_auxiliary_loss
        from stylizer.util import vits_kl_loss
        torch.set_num_threads(2)
        model = SoftVcStylizer(latent_dim=16, decoder_max_loops=1, train_formant=False)
        values = (torch.randn(2, 7, 768), torch.randn(2, 7, 1280), torch.randn(2, 7, 256),
                  torch.randn(2, 192), torch.randn(2, 7, 128), None, None)
        outputs, aux = _stylizer_forward_with_decoder_aux(model, *values)
        self.assertEqual(outputs[0].shape, (2, 7, 768))
        loss = outputs[0].square().mean() + _decoder_auxiliary_loss(DotDict({"train": {}}), aux, outputs[0])
        loss = loss + vits_kl_loss(outputs[5], outputs[4], outputs[1], outputs[2], outputs[11])
        loss = loss + 0.5 * vits_kl_loss(outputs[7], outputs[2], outputs[3], outputs[4], outputs[8])
        loss.backward()
        for parameter in (model.decoder_input_proj.weight_v, model.prior_encoder.pre.weight,
                          model.decoder_exit.content_reconstruction[-1].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == "__main__":
    unittest.main()
