"""Preprocess upstream or SoftVcStylizer features with per-worker model initialization."""
import argparse
import concurrent.futures
import os
from pathlib import Path
import random
import re
import shutil

import librosa
import numpy as np
import torch
import torch.multiprocessing as mp
from tqdm import tqdm

from ddsp.vocoder import F0_Extractor, Volume_Extractor, Units_Encoder
from frontend import apply_overrides, auxiliary_encoders, options
from logger import utils
from reflow.vocoder import Vocoder

_worker = {}


def parse_args(args=None, namespace=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-d", "--device", default=None)
    parser.add_argument("-j", "--workers", type=int, default=2,
                        help="Worker processes; default 2 upstream, 1 with the multi-encoder frontend")
    parser.add_argument("--store_dtype", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--frontend", action="store_true", help="Extract SoftVcStylizer auxiliary features and speaker embeddings")
    return parser.parse_args(args=args, namespace=namespace)


def speaker_model_input(audio, sample_rate, device):
    """Resample only the speaker-model input; do not change training-rate audio."""
    from torchaudio.functional import resample

    waveform = torch.as_tensor(audio, dtype=torch.float32, device=device).reshape(1, -1)
    if sample_rate != 16000:
        waveform = resample(waveform, sample_rate, 16000)
    return waveform


def load_extractors(args, sample_rate=None, hop_size=None, device="cuda"):
    sample_rate = sample_rate or args.data.sampling_rate
    hop_size = hop_size or args.data.block_size
    pitch = F0_Extractor(args.data.f0_extractor, sample_rate, hop_size, args.data.f0_min, args.data.f0_max)
    volume = Volume_Extractor(hop_size, args.data.volume_smooth_size)
    mel = Vocoder(args.vocoder.type, args.vocoder.ckpt, device=device)
    if mel.vocoder_sample_rate != sample_rate or mel.vocoder_hop_size != hop_size:
        raise ValueError("Vocoder sample rate/hop must match data.sampling_rate/data.block_size")
    units = Units_Encoder(args.data.encoder, args.data.encoder_ckpt,
                          args.data.encoder_sample_rate, args.data.encoder_hop_size,
                          cnhubertsoft_gate=args.data.get("cnhubertsoft_gate", 10), device=device)
    return pitch, volume, mel, units


def _worker_init(args, device, store_dtype):
    global _worker
    args = utils.DotDict(args)
    pitch, volume, mel, units = load_extractors(args, device=device)
    _worker = dict(pitch=pitch, volume=volume, mel=mel, units=units,
                   device=device, dtype=store_dtype, frontend=bool(options(args).get("enabled")))
    if _worker["frontend"]:
        from speechbrain.inference.classifiers import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy

        settings = options(args)
        whisper, hubert = auxiliary_encoders(args, device)
        speaker_device = torch.device(device)
        if speaker_device.type == "cuda" and speaker_device.index is None:
            speaker_device = torch.device("cuda", torch.cuda.current_device())
        classifier = EncoderClassifier.from_hparams(
            source=settings.get("speaker_encoder", "speechbrain/spkrec-ecapa-voxceleb"),
            savedir=settings.get("speaker_encoder_cache", "cache/speaker_encoder"),
            run_opts={"device": str(speaker_device)},
            local_strategy=LocalStrategy.COPY,
        )
        _worker.update(whisper=whisper, hubert=hubert, speaker=classifier)


def _dtype_cast(array, dtype):
    return np.ascontiguousarray(array, dtype=dtype)


@torch.no_grad()
def _process_file(file, path, sample_rate, hop_size, use_pitch_aug, extensions):
    source = Path(path) / "audio" / file
    audio, _ = librosa.load(source, sr=sample_rate, mono=True)
    device = _worker["device"]
    waveform = torch.from_numpy(audio).float().to(device).unsqueeze(0)
    f0 = _worker["pitch"].extract(audio, uv_interp=False, device=device)
    unvoiced = f0 == 0
    if not np.any(~unvoiced):
        destination = Path(path) / "skip" / file
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(destination))
        print(f"F0 extraction failed; moved to {destination}")
        return None
    f0[unvoiced] = np.interp(np.where(unvoiced)[0], np.where(~unvoiced)[0], f0[~unvoiced])
    maximum = float(waveform.abs().max()) + 1e-5
    volume_shift = random.uniform(-1, min(1, np.log10(1 / maximum)))
    keyshift = random.uniform(-5, 5) if use_pitch_aug else 0
    amplitude = 10 ** volume_shift

    def array(tensor):
        return tensor.squeeze(0).float().cpu().numpy()

    features = {
        "units": array(_worker["units"].encode(waveform, sample_rate, hop_size)),
        "mel": array(_worker["mel"].extract(waveform, sample_rate)),
        "aug_mel": array(_worker["mel"].extract(waveform * amplitude, sample_rate, keyshift=keyshift)),
        "f0": f0,
        "volume": _worker["volume"].extract(audio),
        "aug_vol": _worker["volume"].extract(audio * amplitude),
    }
    if _worker["frontend"]:
        for key, encoder in (("whisper_units", _worker["whisper"]), ("hubert_units", _worker["hubert"])):
            features[key] = array(encoder.encode(waveform, sample_rate, hop_size))
        speaker_audio = speaker_model_input(audio, sample_rate, device)
        features["speaker"] = _worker["speaker"].encode_batch(speaker_audio).squeeze().float().cpu().numpy()
        if keyshift:
            shifted = librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=keyshift)
            shifted = np.pad(shifted[:len(audio)], (0, max(0, len(audio) - len(shifted))))
            shifted_waveform = torch.from_numpy(shifted).float().to(device).unsqueeze(0)
            for key, encoder in (("units_aug", _worker["units"]),
                                 ("whisper_units_aug", _worker["whisper"]),
                                 ("hubert_units_aug", _worker["hubert"])):
                features[key] = array(encoder.encode(shifted_waveform, sample_rate, hop_size))
        else:
            for key in ("units", "whisper_units", "hubert_units"):
                features["units_aug" if key == "units" else key + "_aug"] = features[key]
    for key, value in features.items():
        destination = Path(path) / key / (file + ".npy")
        destination.parent.mkdir(parents=True, exist_ok=True)
        dtype = np.float32 if key in ("f0", "volume", "aug_vol", "speaker") else _worker["dtype"]
        np.save(destination, _dtype_cast(value, dtype))
    return keyshift


def save_speaker_means(path, files, n_spk=1, speaker_means=None):
    """Save FP32 speaker means, computing them only when none are supplied."""
    if speaker_means is None:
        totals, counts = {}, {}
        for file in files:
            speaker_id = 1
            if n_spk is not None and n_spk > 1:
                prefix = re.split(r"_|\-", os.path.dirname(file), 2)[0]
                speaker_id = int(prefix) if prefix.isdigit() else 0
                if not 1 <= speaker_id <= n_spk:
                    raise ValueError(f"Invalid speaker directory for {file}; expected 1..{n_spk}")
            speaker = np.load(Path(path) / "speaker" / (file + ".npy")).astype(np.float32).squeeze()
            if speaker_id not in totals:
                totals[speaker_id] = speaker
            else:
                totals[speaker_id] += speaker
            counts[speaker_id] = counts.get(speaker_id, 0) + 1
        speaker_means = {speaker_id: total / counts[speaker_id] for speaker_id, total in totals.items()}
    destination = Path(path) / "speaker_means"
    destination.mkdir(exist_ok=True)
    for speaker_id, mean in speaker_means.items():
        np.save(destination / f"{speaker_id}.npy", mean)
    return speaker_means


def preprocess(path, args, sample_rate=None, hop_size=None, device="cuda",
               use_pitch_aug=False, extensions=("wav",), workers=1, store_dtype=None, speaker_means=None):
    sample_rate = sample_rate or args.data.sampling_rate
    hop_size = hop_size or args.data.block_size
    files = utils.traverse_dir(str(Path(path) / "audio"), extensions,
                               is_pure=True, is_sort=True, is_ext=True)
    if not files:
        print(f"No audio files found under {path}/audio")
        return speaker_means if speaker_means is not None else {}
    if workers < 1:
        raise ValueError("workers must be at least 1")
    store_dtype = np.float32 if store_dtype is None else store_dtype
    initializer_args = (dict(args), device, store_dtype)
    parameters = (path, sample_rate, hop_size, use_pitch_aug, extensions)
    if workers == 1:
        _worker_init(*initializer_args)
        results = [_process_file(file, *parameters) for file in tqdm(files)]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers,
                mp_context=mp.get_context("spawn"), initializer=_worker_init,
                initargs=initializer_args) as executor:
            futures = [executor.submit(_process_file, file, *parameters) for file in files]
            results = [future.result() for future in tqdm(futures)]
    pitch_shifts = {file: shift for file, shift in zip(files, results) if shift is not None}
    np.save(Path(path) / "pitch_aug_dict.npy", pitch_shifts)
    if options(args).get("enabled"):
        return save_speaker_means(path, pitch_shifts, args.model.n_spk, speaker_means)


def main():
    command = parse_args()
    args = utils.load_config(command.config)
    device = command.device or ("cuda" if torch.cuda.is_available() else "cpu")
    workers = command.workers if command.workers is not None else 2
    store_dtype = np.float16 if command.store_dtype == "fp16" else np.float32
    speaker_means = None
    for path, augment in ((args.data.train_path, bool(args.model.use_pitch_aug)),
                          (args.data.valid_path, False)):
        speaker_means = preprocess(path, args, device=device, workers=workers, store_dtype=store_dtype,
                                   use_pitch_aug=augment, extensions=args.data.extensions,
                                   speaker_means=speaker_means)


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
