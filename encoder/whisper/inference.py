# Copyright OpenAI. MIT License; see LICENSE in this directory.
"""Audio-only Whisper large-v2 PPG features."""
import torch
import torch.nn.functional as F

from .audio import log_mel_spectrogram
from .model import AudioEncoder, ModelDimensions


def load_ppg_model(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    dims = ModelDimensions(**checkpoint["dims"])
    layers = dims.n_audio_layer - dims.n_audio_layer // 4
    encoder = AudioEncoder(dims.n_mels, dims.n_audio_ctx, dims.n_audio_state, dims.n_audio_head, layers)
    state = {}
    for key, value in checkpoint["model_state_dict"].items():
        if not key.startswith("encoder."):
            continue
        key = key[len("encoder."):]
        if key.startswith("blocks.") and int(key.split(".")[1]) >= layers:
            continue
        state[key] = value
    encoder.load_state_dict(state, strict=True)
    encoder = encoder.eval().to(device)
    return encoder.half() if torch.device(device).type == "cuda" else encoder.float()


@torch.no_grad()
def pred_ppg(encoder, audio, device):
    audio = audio.reshape(-1)
    outputs = []
    dtype = next(encoder.parameters()).dtype
    for start in range(0, audio.numel(), 15 * 16000):
        chunk = audio[start:start + 15 * 16000]
        frame_count = max(1, chunk.numel() // 320)
        if chunk.numel() < 400:
            chunk = F.pad(chunk, (0, 400 - chunk.numel()))
        mel = log_mel_spectrogram(chunk).to(device=device, dtype=dtype).unsqueeze(0)
        # Preserve the original PPG feature augmentation.
        mel = mel + torch.randn_like(mel) * 0.1
        outputs.append(encoder(mel)[:, :frame_count].float())
    return torch.cat(outputs, dim=1)
