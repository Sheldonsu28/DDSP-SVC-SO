# Copyright OpenAI. MIT License; see LICENSE in this directory.
"""Whisper log-mel features used by PPG extraction."""
from functools import lru_cache

import torch
from librosa.filters import mel as librosa_mel

SAMPLE_RATE = 16000
N_FFT = 400
HOP_LENGTH = 160
N_MELS = 80


@lru_cache(maxsize=None)
def mel_filters(device, n_mels=N_MELS):
    if n_mels != N_MELS:
        raise ValueError("Whisper large-v2 PPG expects 80 mel bins")
    return torch.from_numpy(librosa_mel(sr=SAMPLE_RATE, n_fft=N_FFT, n_mels=n_mels)).to(device)


def log_mel_spectrogram(audio):
    window = torch.hann_window(N_FFT, device=audio.device)
    spectrum = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = spectrum[..., :-1].abs().square()
    mel = mel_filters(audio.device) @ magnitudes
    log_mel = mel.clamp_min(1e-10).log10()
    log_mel = torch.maximum(log_mel, log_mel.max() - 8.0)
    return (log_mel + 4.0) / 4.0
