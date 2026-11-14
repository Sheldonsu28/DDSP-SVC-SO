"""Shared resource settings for the SoftVcStylizer preprocessing/training/inference path."""
from pathlib import Path
import pickle

import numpy as np
import torch
import yaml

from logger.utils import DotDict

FEATURE_STREAMS = {
    "units": "feature_and_index.pkl",
    "hubert_units": "feature_and_index_hubert.pkl",
    "whisper_units": "feature_and_index_whisper.pkl",
}


def options(args):
    return DotDict(args.get("frontend") or {})


def apply_overrides(args, command):
    settings = dict(options(args))
    for name in ("checkpoint", "speaker_embedding", "index_dir", "contentvec_index", "hubert_index", "whisper_index"):
        value = getattr(command, "frontend_checkpoint" if name == "checkpoint" else name, None)
        if value is not None:
            settings[name] = value
    if getattr(command, "frontend", False) or getattr(command, "frontend_checkpoint", None):
        settings["enabled"] = True
    elif "enabled" not in settings:
        settings["enabled"] = bool(settings.get("checkpoint"))
    if getattr(command, "no_frontend", False):
        settings["enabled"] = False
    args["frontend"] = settings
    return options(args)


def auxiliary_encoders(args, device):
    from ddsp.vocoder import Units_Encoder

    settings = options(args)
    tta = args.data.encoder_hop_size == 160
    whisper = Units_Encoder(
        "whisper-large-pgg-tta2x" if tta else "whisper-large-pgg",
        settings.get("whisper_ckpt", "pretrain/ppg-large/large-v2.pt"),
        16000, 160 if tta else 320, device=device,
    )
    hubert = Units_Encoder(
        "hubertsofttta2x" if tta else "hubertsoftOri",
        settings.get("hubert_ckpt", "pretrain/hubert/hubert-soft.pt"),
        16000, 160 if tta else 320, device=device,
    )
    return whisper, hubert


def load_frontend(checkpoint_path, device="cpu", model_options=None):
    from stylizer.stylizer import SoftVcStylizer

    checkpoint_path = Path(checkpoint_path)
    config_path = checkpoint_path.with_name("config.yaml")
    model_settings = {}
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as stream:
            saved = yaml.safe_load(stream) or {}
        model_settings.update((saved.get("frontend") or {}).get("model") or {})
    model_settings.update(model_options or {})
    model = SoftVcStylizer(**model_settings).to(device)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.eval()


def load_indexes(settings, speaker_id):
    paths = {
        "units": settings.get("contentvec_index"),
        "hubert_units": settings.get("hubert_index"),
        "whisper_units": settings.get("whisper_index"),
    }
    result = {}
    for stream, filename in FEATURE_STREAMS.items():
        path = paths[stream]
        if not path and settings.get("index_dir"):
            path = str(Path(settings["index_dir"]) / filename)
        if not path:
            raise ValueError(f"Set frontend.index_dir or an explicit {stream} index path")
        with open(path, "rb") as file:
            indexes = pickle.load(file)
        if speaker_id not in indexes:
            raise ValueError(f"Speaker {speaker_id} is absent from index {path}")
        index = indexes[speaker_id]
        if not index.ntotal:
            raise ValueError(f"Empty feature index: {path}")
        result[stream] = (index, index.reconstruct_n(0, index.ntotal))
    return result


def retrieve(features, index_data, neighbors):
    """Preserve the source's diagonal standard-deviation distribution mapping."""
    from utils import map_normal_diagonal

    index, vectors = index_data
    if features.shape[-1] != index.d:
        raise ValueError(f"Feature dimension {features.shape[-1]} does not match index dimension {index.d}")
    query = np.ascontiguousarray(features.squeeze(0).float().cpu().numpy(), dtype=np.float32)
    _, indices = index.search(query, min(neighbors, index.ntotal))
    retrieved = torch.from_numpy(vectors[indices].mean(axis=1)).to(features)
    source_std, source_mean = torch.std_mean(features, dim=1)
    target_std, target_mean = torch.std_mean(retrieved, dim=0)
    return map_normal_diagonal(features, source_mean, source_std, target_mean, target_std)
