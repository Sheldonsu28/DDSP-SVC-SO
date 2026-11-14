"""Load the DDSP synthesizer from a DDSP-SVC 6.3 checkpoint."""

from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from .vocoder import CombSubSuperFast, DotDict


class DDSPOnlyUnit2Wav(torch.nn.Module):
    """DDSP-only module with the ``Unit2Wav`` forward interface.

    Training returns ``(ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav,
    ddsp_mel)``. Reflow and band losses are zero; DDSP loss is mel MSE, as in
    ``Unit2Wav``. Supply a differentiable vocoder's ``extract`` method and
    ``gt_spec`` to compute it. No gradients are disabled by this wrapper.

    Inference returns DDSP mel features, or vocoder-resynthesized audio when
    ``return_wav=True``. Without a vocoder, ``return_wav=True`` returns the raw
    DDSP waveform. Reflow sampling arguments are accepted and ignored.
    """

    def __init__(self, ddsp_model):
        super().__init__()
        self.ddsp_model = ddsp_model
        self.sampling_rate = int(ddsp_model.sampling_rate)
        self.block_size = int(ddsp_model.block_size)

    def extract_mel(self, ddsp_wav, vocoder):
        return vocoder.extract(ddsp_wav)

    def infer_audio(self, mel, f0, vocoder):
        return vocoder.infer(mel, f0)

    def forward(self, units, f0, volume, spk_id=None, spk_mix_dict=None,
                aug_shift=None, vocoder=None, gt_spec=None, infer=True,
                return_wav=False, infer_step=10, method='euler', t_start=0.0,
                silence_front=0, use_tqdm=False):
        if not infer and (vocoder is None or gt_spec is None):
            raise ValueError('Training requires vocoder and gt_spec for DDSP mel loss')

        if aug_shift is not None:
            # Frontend heads and validation can return a scalar, one value per
            # example, or one value per frame. Preserve those meanings without
            # accidentally broadcasting the batch axis over the time axis.
            aug_shift = torch.as_tensor(aug_shift, device=units.device, dtype=units.dtype)
            if aug_shift.ndim == 0:
                aug_shift = aug_shift.reshape(1, 1, 1)
            elif aug_shift.ndim == 1:
                aug_shift = aug_shift[:, None, None]
            elif aug_shift.ndim == 2:
                aug_shift = aug_shift.unsqueeze(-1)
            aug_shift = torch.broadcast_to(aug_shift, (*units.shape[:2], 1))

        ddsp_wav, _ = self.ddsp_model(
            units, f0, volume, spk_id=spk_id, spk_mix_dict=spk_mix_dict,
            aug_shift=aug_shift, infer=infer,
        )
        start_frame = int(silence_front * self.sampling_rate / self.block_size)
        trimmed_wav = ddsp_wav[:, start_frame * self.block_size:]
        ddsp_mel = self.extract_mel(trimmed_wav, vocoder) if vocoder is not None else None

        if not infer:
            ddsp_loss = F.mse_loss(ddsp_mel, gt_spec)
            # Keep placeholder losses on the right device, dtype and graph.
            reflow_loss = ddsp_loss * 0.0
            ddsp_band_loss = ddsp_loss * 0.0
            return ddsp_loss, reflow_loss, ddsp_band_loss, ddsp_wav, ddsp_mel

        if return_wav and vocoder is None:
            return trimmed_wav
        if ddsp_mel is None:
            ddsp_mel = gt_spec
        if ddsp_mel is None:
            raise ValueError('Mel inference requires vocoder or gt_spec')
        if return_wav:
            return self.infer_audio(ddsp_mel, f0[:, -ddsp_mel.shape[1]:], vocoder)
        return ddsp_mel


def load_ddsp_model(model_path, device='cpu', config_path=None, wrapped=True):
    """Return ``(model, args)`` without constructing reflow or a vocoder.

    Args:
        model_path: An experiment folder or a DDSP-SVC 6.3 checkpoint containing
            a ``model`` state dict with ``ddsp_model.*`` keys. For a folder,
            selects ``model_<step>.pt`` with the largest numeric step, following
            ``logger.utils.load_model``'s naming convention.
        device: Device for the returned model (a string or ``torch.device``).
        config_path: Matching YAML config or a folder containing ``config.yaml``.
            Defaults to ``config.yaml`` beside the checkpoint. Only its ``data``
            and ``model`` sections are used.
        wrapped: Return a ``DDSPOnlyUnit2Wav`` with the normal ``Unit2Wav``
            call/return interface. Set False to return bare ``CombSubSuperFast``
            whose forward call returns ``(waveform, hidden)``.

    The checkpoint is read on CPU, so reflow and optimizer tensors are never
    transferred to the requested device. DDSP weights are loaded strictly and
    the returned model is in evaluation mode; call ``model.train()`` to train.
    The raw synthesizer is available as ``model.ddsp_model`` when wrapped.
    """
    model_path = Path(model_path)
    if model_path.is_dir():
        checkpoints = [
            (int(path.stem[len('model_'):]), path)
            for path in model_path.glob('model_*.pt')
            if path.is_file() and path.stem[len('model_'):].isdigit()
        ]
        if not checkpoints:
            raise FileNotFoundError(
                f'No model_<step>.pt checkpoints found in {model_path}'
            )
        model_path = max(checkpoints, key=lambda checkpoint: checkpoint[0])[1]
    config_path = (
        Path(config_path) if config_path is not None
        else model_path.with_name('config.yaml')
    )
    if config_path.is_dir():
        config_path = config_path / 'config.yaml'
    with config_path.open('r', encoding='utf-8') as config:
        args = DotDict(yaml.safe_load(config))

    if args.model.type != 'RectifiedFlow':
        raise ValueError(f'Unknown model type: {args.model.type}')

    print(' [*] restoring DDSP model from', model_path)
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    prefix = 'ddsp_model.'
    state_dict = {
        key[len(prefix):]: value
        for key, value in checkpoint['model'].items()
        if key.startswith(prefix)
    }
    if not state_dict:
        raise ValueError(f'No ddsp_model.* weights found in {model_path}')
    del checkpoint

    model = CombSubSuperFast(
        sampling_rate=args.data.sampling_rate,
        block_size=args.data.block_size,
        win_length=args.model.win_length,
        n_unit=args.data.encoder_out_channels,
        n_spk=args.model.n_spk,
        num_layers=(
            args.model.n_aux_layers if args.model.n_aux_layers is not None else 3
        ),
        dim_model=(
            args.model.n_aux_chans if args.model.n_aux_chans is not None else 256
        ),
        use_norm=args.model.use_norm,
        use_attention=args.model.use_attention,
        use_pitch_aug=args.model.use_pitch_aug,
    )
    model.load_state_dict(state_dict, strict=True)
    if wrapped:
        model = DDSPOnlyUnit2Wav(model)
    model.to(device)
    model.eval()
    return model, args
