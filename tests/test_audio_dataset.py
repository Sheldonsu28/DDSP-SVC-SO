import unittest
from unittest.mock import patch

import torch
from torch.utils.data import DataLoader

from reflow.data_loaders import AudioDataset


class AudioDatasetBatchCropTests(unittest.TestCase):
    def make_dataset(self, use_short=True, whole_audio=False):
        # Exercise real cropping and collation without loading training files.
        dataset = AudioDataset.__new__(AudioDataset)
        dataset.hop_size = 512
        dataset.sample_rate = 44100
        dataset.crop_len = 8
        dataset.short_len = 3
        dataset.use_short = use_short
        dataset.whole_audio = whole_audio
        dataset.use_aug = False
        dataset.aug_embd = False
        dataset.load_audio = True
        dataset.train_frontend = True
        dataset.paths = [f'sample_{i}.wav' for i in range(5)]
        dataset.data_buffer = {}
        for name in dataset.paths:
            frames = torch.arange(12, dtype=torch.float32).unsqueeze(-1)
            dataset.data_buffer[name] = {
                'frame_len': 12,
                **{key: frames.clone() for key in (
                    'mel', 'units', 'units_w', 'units_h',
                    'f0', 'volume',
                )},
                'spk_id': torch.tensor([1]),
                'spk_embd': torch.ones(4),
                'gt_audio': torch.arange(12 * 512, dtype=torch.float32),
            }
        return dataset

    def assert_batch_crop(self, batch, batch_size, crop_len):
        expected = torch.arange(1, 1 + crop_len, dtype=torch.float32)
        expected = expected[None, :, None].expand(batch_size, -1, -1)
        for key in ('mel', 'units', 'units_ori', 'units_w',
                    'units_h', 'f0', 'volume'):
            torch.testing.assert_close(batch[key], expected)
        expected_audio = torch.arange(512, 512 + crop_len * 512, dtype=torch.float32)
        torch.testing.assert_close(
            batch['gt_audio'], expected_audio[None, :].expand(batch_size, -1),
        )

    def test_batches_choose_one_length_and_keep_all_features_aligned(self):
        dataset = self.make_dataset()
        batches = iter(DataLoader(dataset, batch_size=2))
        with patch('reflow.data_loaders.random.randint', return_value=1):
            for short, batch_size in ((True, 2), (False, 2), (True, 1)):
                with self.subTest(short=short, batch_size=batch_size):
                    with patch('reflow.data_loaders.random.choice', return_value=short):
                        batch = next(batches)
                    self.assert_batch_crop(batch, batch_size, 3 if short else 8)
        self.assertEqual(dataset.crop_len, 8)

    def test_disabled_short_crops_keep_normal_length(self):
        dataset = self.make_dataset(use_short=False)
        with patch('reflow.data_loaders.random.choice', return_value=True), \
                patch('reflow.data_loaders.random.randint', return_value=1):
            self.assert_batch_crop(next(iter(DataLoader(dataset, batch_size=2))), 2, 8)

    def test_whole_audio_keeps_full_length(self):
        dataset = self.make_dataset(whole_audio=True)
        with patch('reflow.data_loaders.random.choice', return_value=True):
            batch = next(iter(DataLoader(dataset, batch_size=2)))
        self.assertEqual(batch['mel'].shape, (2, 12, 1))
        torch.testing.assert_close(batch['mel'][0], dataset.data_buffer['sample_0.wav']['mel'])

    def test_skipping_too_short_audio_preserves_batch_crop(self):
        dataset = self.make_dataset()
        dataset.data_buffer['sample_0.wav']['frame_len'] = 2
        dataset.data_buffer['sample_1.wav']['frame_len'] = 4
        with patch('reflow.data_loaders.random.choice', return_value=True), \
                patch('reflow.data_loaders.random.randint', return_value=1):
            batch = next(iter(DataLoader(dataset, batch_size=2)))
        self.assertEqual(batch['name_ext'], ['sample_1.wav', 'sample_1.wav'])
        self.assert_batch_crop(batch, 2, 3)

    def test_worker_batches_stack_with_consistent_lengths(self):
        dataset = self.make_dataset()
        loader = DataLoader(dataset, batch_size=2, num_workers=2, persistent_workers=True)
        for _ in range(2):
            for batch in loader:
                length = batch['mel'].shape[1]
                self.assertIn(length, (dataset.short_len, dataset.crop_len))
                self.assertEqual(batch['units'].shape[1], length)
                self.assertEqual(batch['gt_audio'].shape[1], length * 512)


if __name__ == '__main__':
    unittest.main()
