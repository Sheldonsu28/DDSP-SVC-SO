from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf
import torch

from logger.utils import DotDict
from main_reflow import load_speaker_embedding
from preprocess import preprocess, save_speaker_means
from reflow.data_loaders import AudioDataset, get_data_loaders


def write_dataset(root, speakers):
    root = Path(root)
    shifts = {}
    for speaker_id, embeddings in speakers.items():
        for index, embedding in enumerate(embeddings):
            name = Path(str(speaker_id)) / f"clip_{index}.wav"
            audio_path = root / "audio" / name
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(audio_path, np.zeros(4 * 512, dtype=np.float32), 44100)
            features = {key: np.zeros((4, channels), dtype=np.float32) for key, channels in
                        (("units", 768), ("whisper_units", 1280), ("hubert_units", 256), ("mel", 128), ("aug_mel", 128))}
            features.update(f0=np.full(4, 220, dtype=np.float32),
                            volume=np.ones(4, dtype=np.float32), aug_vol=np.ones(4, dtype=np.float32),
                            speaker=np.asarray(embedding, dtype=np.float32))
            for key, values in features.items():
                path = root / key / (str(name) + ".npy")
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, values)
            shifts[str(name)] = 0
    np.save(root / "pitch_aug_dict.npy", shifts)
    save_speaker_means(root, shifts, n_spk=2)


def loader_args(root):
    return DotDict(dict(
        data=dict(train_path=str(root / "train"), valid_path=str(root / "val"),
                  sampling_rate=44100, block_size=512, duration=0.02, extensions=["wav"]),
        model=dict(n_spk=2),
        train=dict(cache_all_data=False, cache_device="cpu", cache_fp16=False, batch_size=2, num_workers=0),
        frontend=dict(speaker_embedding=str(root / "unused_global_mean.npy")),
    ))


class SpeakerMeanTests(unittest.TestCase):
    def test_each_speaker_uses_its_mean_in_cached_and_uncached_modes(self):
        with tempfile.TemporaryDirectory() as directory:
            write_dataset(directory, {1: [[1, 0], [0, 1]], 2: [[-1, 0], [0, -1]]})
            expected = {1: torch.tensor([0.5, 0.5]), 2: torch.tensor([-0.5, -0.5])}
            for cache in (False, True):
                with self.subTest(cache=cache):
                    with patch("reflow.data_loaders.np.load", wraps=np.load) as load:
                        dataset = AudioDataset(directory, 0.02, 512, 44100, n_spk=2,
                            train_frontend=True, whole_audio=True, load_all_data=cache, fp16=True)
                        for _ in range(2):
                            for index in range(len(dataset)):
                                item = dataset[index]
                                speaker_id = item["spk_id"].item()
                                torch.testing.assert_close(item["spk_embd"], expected[speaker_id])
                                self.assertEqual(item["spk_embd"].dtype, torch.float32)
                    loaded = [Path(call.args[0]) for call in load.call_args_list]
                    self.assertCountEqual(
                        [path for path in loaded if path.parent.name == "speaker_means"],
                        [Path(directory) / "speaker_means" / f"{speaker_id}.npy" for speaker_id in expected])
                    self.assertFalse(any(path.is_relative_to(Path(directory) / "speaker") for path in loaded))

    def test_preprocess_saves_only_successful_clips_in_fp32(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root, {1: [[1, 0], [0, 1]], 2: [[-1, 0], [0, -1]]})
            args = DotDict(dict(data=dict(sampling_rate=44100, block_size=512),
                                model=dict(n_spk=2), frontend=dict(enabled=True)))
            with patch("preprocess._worker_init"), patch("preprocess._process_file", side_effect=[0, None, 0, 0]):
                preprocess(root, args, device="cpu", workers=1)
            for speaker_id, expected in {1: [1, 0], 2: [-0.5, -0.5]}.items():
                mean = np.load(root / "speaker_means" / f"{speaker_id}.npy")
                np.testing.assert_array_equal(mean, expected)
                self.assertEqual(mean.dtype, np.float32)

    def test_inference_automatically_loads_the_selected_mean_from_train_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root, {1: [[1, 0]], 2: [[0, 1]]})
            with patch("main_reflow.np.load", wraps=np.load) as load:
                embedding = load_speaker_embedding(None, 2, train_path=root)
                load.assert_called_once_with(root / "speaker_means" / "2.npy")
            torch.testing.assert_close(embedding, torch.tensor([0.0, 1.0]))
            with self.assertRaises(FileNotFoundError):
                load_speaker_embedding(None, 3, train_path=root)

    def test_inference_selects_mean_by_speaker_id_and_accepts_an_explicit_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root, {1: [[1, 0]], 2: [[0, 1]]})
            means = root / "speaker_means"
            with patch("main_reflow.np.load", wraps=np.load) as load:
                embedding = load_speaker_embedding(means, 2, train_path=root / 'unused')
                load.assert_called_once_with(means / "2.npy")
            torch.testing.assert_close(embedding, torch.tensor([0.0, 1.0]))
            torch.testing.assert_close(load_speaker_embedding(means / "1.npy", 2, train_path=root / "unused"),
                                       torch.tensor([1.0, 0.0]))
            with self.assertRaises(FileNotFoundError):
                load_speaker_embedding(means, 3)

    def test_validation_reuses_training_means_and_ignores_global_inference_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root / "train", {1: [[1, 0], [0, 1]], 2: [[-1, 0], [0, -1]]})
            write_dataset(root / "val", {1: [[100, 100]], 2: [[900, 900]]})
            train, validation = get_data_loaders(loader_args(root), frontend=True, train_aug=False)
            expected = {1: torch.tensor([0.5, 0.5]), 2: torch.tensor([-0.5, -0.5])}
            for dataset in (train.dataset, validation.dataset):
                for index in range(len(dataset)):
                    item = dataset[index]
                    torch.testing.assert_close(item["spk_embd"], expected[item["spk_id"].item()])

    def test_validation_rejects_a_speaker_without_training_embeddings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_dataset(root / "train", {1: [[1, 0]]})
            write_dataset(root / "val", {2: [[0, 1]]})
            with self.assertRaisesRegex(ValueError, "No training speaker mean for speaker 2"):
                get_data_loaders(loader_args(root), frontend=True, train_aug=False)


if __name__ == "__main__":
    unittest.main()
