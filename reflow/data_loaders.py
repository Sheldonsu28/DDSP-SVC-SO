"""Aligned upstream and SoftVcStylizer frontend training datasets."""
import concurrent.futures
import os
import random
import re
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from logger.utils import traverse_dir


def get_npy_shape(file_path):
    with open(file_path, "rb") as stream:
        version = np.lib.format.read_magic(stream)
        reader = { (1, 0): np.lib.format.read_array_header_1_0,
                   (2, 0): np.lib.format.read_array_header_2_0 }.get(version)
        if reader is None:
            raise ValueError(f"Unsupported NumPy file version: {version}")
        return reader(stream)[0]


def get_data_loaders(args, whole_audio=False, frontend=False, load_audio=False,
                     train_aug=True, aug_embd=False, use_short=False):
    common = dict(hop_size=args.data.block_size, sample_rate=args.data.sampling_rate,
                  load_all_data=args.train.cache_all_data, extensions=args.data.extensions,
                  n_spk=args.model.n_spk, train_frontend=frontend)
    training = AudioDataset(args.data.train_path, waveform_sec=args.data.duration,
                            whole_audio=whole_audio, device=args.train.cache_device,
                            fp16=args.train.cache_fp16, use_aug=train_aug, load_audio=load_audio,
                            aug_embd=aug_embd, use_short=use_short, **common)
    validation = AudioDataset(args.data.valid_path, waveform_sec=2, whole_audio=True,
                              load_audio=True, fp16=False,
                              speaker_means=training.speaker_means if frontend else None, **common)
    workers = args.train.num_workers if args.train.cache_device == "cpu" else 0
    loader_train = DataLoader(training, batch_size=1 if whole_audio else args.train.batch_size,
                              shuffle=True, num_workers=workers, persistent_workers=workers > 0,
                              pin_memory=args.train.cache_device == "cpu")
    loader_valid = DataLoader(validation, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    return loader_train, loader_valid


class AudioDataset(Dataset):
    def __init__(self, path_root, waveform_sec, hop_size, sample_rate,
                 load_all_data=True, whole_audio=False, extensions=("wav",), n_spk=1,
                 device="cpu", fp16=False, use_aug=False, train_frontend=False,
                 load_audio=False, aug_embd=False,
                 use_short=False, speaker_means=None):
        self.path_root = str(path_root)
        self.hop_size = int(hop_size)
        self.sample_rate = int(sample_rate)
        self.crop_len = max(1, int(waveform_sec * sample_rate / hop_size))
        self.short_len = max(1, int(0.4 * sample_rate / hop_size))
        self.whole_audio = whole_audio
        self.use_aug = use_aug
        self.train_frontend = train_frontend
        self.load_audio = load_audio
        self.aug_embd = aug_embd
        self.use_short = use_short
        self.device = device
        self.fp16 = fp16
        self.speaker_means = {} if speaker_means is None else {
            speaker_id: mean.to(device=device, dtype=torch.float32)
            for speaker_id, mean in speaker_means.items()}
        self.paths = traverse_dir(os.path.join(self.path_root, "audio"), extensions,
                                  is_pure=True, is_sort=True, is_ext=True)
        if not self.paths:
            raise ValueError(f"No audio files found under {self.path_root}/audio")
        pitch_path = Path(self.path_root) / "pitch_aug_dict.npy"
        self.pitch_aug_dict = np.load(pitch_path, allow_pickle=True).item() if pitch_path.is_file() else {}
        self.data_buffer = {}
        self.feature_dirs = {"mel": "mel", "aug_mel": "aug_mel", "units": "units"}
        
        if train_frontend:
            self.feature_dirs.update(units_w="whisper_units", units_h="hubert_units", spk_embd="speaker")
            
        if aug_embd:
            self.feature_dirs["units_aug"] = "units_aug"
            if train_frontend:
                self.feature_dirs.update(units_w_aug="whisper_units_aug", units_h_aug="hubert_units_aug")
                
        def _load_single_file(name_ext):
            data = {}
            lengths = []
            
            for key in ("f0", "volume", "aug_vol"):
                array = np.load(self._path(key, name_ext))
                lengths.append(len(array))
                data[key] = torch.from_numpy(array).float().reshape(-1, 1).to(device)
            for key, folder in self.feature_dirs.items():
                file_path = self._path(folder, name_ext)
                lengths.append(get_npy_shape(file_path)[0])
                if load_all_data:
                    data[key] = self._tensor(np.load(file_path))
            data["frame_len"] = min(lengths)
            speaker_id = 1
            if n_spk is not None and n_spk > 1:
                prefix = re.split(r"_|\-", os.path.dirname(name_ext), 2)[0]
                speaker_id = int(prefix) if prefix.isdigit() else 0
                if not 1 <= speaker_id <= n_spk:
                    raise ValueError(f"Invalid speaker directory for {name_ext}; expected 1..{n_spk}")
            data["spk_id"] = torch.tensor([speaker_id], device=device, dtype=torch.long)
            if load_audio:
                waveform, _ = librosa.load(Path(self.path_root) / "audio" / name_ext, sr=sample_rate)
                data["gt_audio"] = torch.from_numpy(waveform).float().to(device)
            return name_ext, data

        max_workers = max(min(32, os.cpu_count() or 1), 4)
        print(f"Using {max_workers} workers for parallel data loading")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_name = {executor.submit(_load_single_file, name_ext): name_ext for name_ext in self.paths}
            for future in tqdm(concurrent.futures.as_completed(future_to_name), total=len(self.paths), desc="Loading data"):
                try:
                    name_ext, data = future.result()
                    self.data_buffer[name_ext] = data
                except Exception as error:
                    print(f"Error loading {future_to_name[future]}: {error}")
                    raise
        if train_frontend:
            # Load each mean once on the main thread, after per-file loading finishes.
            for data in self.data_buffer.values():
                speaker_id = data["spk_id"].item()
                if speaker_means is None and speaker_id not in self.speaker_means:
                    mean_path = Path(self.path_root) / "speaker_means" / f"{speaker_id}.npy"
                    self.speaker_means[speaker_id] = torch.from_numpy(np.load(mean_path)).float().to(device).squeeze()
                if speaker_id not in self.speaker_means:
                    raise ValueError(f"No training speaker mean for speaker {speaker_id}")
                data["spk_embd"] = self.speaker_means[speaker_id] if torch.cosine_similarity(data["spk_embd"], self.speaker_means[speaker_id], dim=0) > 0.85 else data["spk_embd"]

    def _path(self, folder, name_ext):
        return str(Path(self.path_root) / folder / (name_ext + ".npy"))

    def _tensor(self, array):
        tensor = torch.from_numpy(np.array(array, copy=True)).to(self.device)
        return tensor.half() if self.fp16 else tensor.float()

    def __getitems__(self, file_indices):
        crop_len = self.crop_len
        if self.use_short and not self.whole_audio and random.choice([True, False]):
            crop_len = self.short_len
        return [self.__getitem__(index, crop_len=crop_len) for index in file_indices]

    def __getitem__(self, file_idx, crop_len=None):
        crop_len = self.crop_len if crop_len is None else crop_len
        for offset in range(len(self.paths)):
            name_ext = self.paths[(file_idx + offset) % len(self.paths)]
            data = self.data_buffer[name_ext]
            if self.whole_audio or data["frame_len"] >= crop_len:
                return self.get_data(name_ext, data, crop_len)
        raise ValueError(f"Every clip is shorter than the requested {crop_len}-frame crop")

    def frame_idx_to_audio(self, embedding_start_idx, embedding_end_idx):
        """Cached features are already aligned to the output hop, not the raw encoder hop."""
        return embedding_start_idx * self.hop_size, embedding_end_idx * self.hop_size

    def get_data(self, name_ext, data_buffer, crop_len=None):
        crop_len = self.crop_len if crop_len is None else crop_len
        start = 0 if self.whole_audio else random.randint(0, data_buffer["frame_len"] - crop_len)
        frames = data_buffer["frame_len"] if self.whole_audio else crop_len
        end = start + frames
        aug = random.choice([True, False]) and self.use_aug
        units_aug = random.choice([True, False]) and self.aug_embd

        def feature(key, folder):
            tensor = data_buffer.get(key)
            if tensor is not None:
                return tensor[start:end]
            array = np.load(self._path(folder, name_ext), mmap_mode="r")
            return self._tensor(array[start:end])

        units_original = feature("units", "units")
        units = feature("units_aug", "units_aug") if units_aug else units_original
        shift = self.pitch_aug_dict[name_ext] if aug else 0
        data = dict(mel=feature("aug_mel" if aug else "mel", "aug_mel" if aug else "mel"),
                    units=units, f0=data_buffer["f0"][start:end] * 2 ** (shift / 12),
                    volume=data_buffer["aug_vol" if aug else "volume"][start:end],
                    spk_id=data_buffer["spk_id"], aug_shift=torch.tensor([[shift]], dtype=torch.float32),
                    name=os.path.splitext(name_ext)[0], name_ext=name_ext)
        if self.train_frontend:
            for key, folder in (("units_w", "whisper_units"), ("units_h", "hubert_units")):
                data[key] = feature(key + "_aug", folder + "_aug") if units_aug else feature(key, folder)
            data["units_ori"] = units_original
            data["spk_embd"] = data_buffer["spk_embd"]
        if self.load_audio:
            audio_start, audio_end = self.frame_idx_to_audio(start, end)
            waveform = data_buffer["gt_audio"][audio_start:audio_end]
            data["gt_audio"] = F.pad(waveform, (0, frames * self.hop_size - waveform.numel()))
        return data

    def __len__(self):
        return len(self.paths)
