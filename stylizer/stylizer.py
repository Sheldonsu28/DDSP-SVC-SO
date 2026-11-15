"""Selected dependencies of the SoftVcStylizer frontend workflow."""

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
from torch.utils.checkpoint import checkpoint
from .modules_grl import GradientReversal
from .wavenet import WN
from .flow import ResidualCouplingBlockVits
from .stlyizer_gen2 import ConformerBlock3, RoformerBlock

class StylizerRefiner(nn.Module):

    def __init__(self, dim=768, num_layers=2, dropout=0.1, use_norm=False, usePrelu=False):
        super().__init__()
        self.norm_layer = nn.LayerNorm(dim)
        layers = []
        for _ in range(num_layers):
            layers.append(nn.Sequential(weight_norm(nn.Linear(dim, dim)) if use_norm else nn.Linear(dim, dim), nn.PReLU() if usePrelu else nn.ReLU(), nn.Dropout(dropout)))
        self.refine = nn.Sequential(*layers)
        self.residual_proj = weight_norm(nn.Linear(dim, dim)) if use_norm else nn.Linear(dim, dim)

    def forward(self, stylized_feats):
        refined = self.refine(stylized_feats)
        residual = self.residual_proj(refined)
        return stylized_feats + residual

class CnnSpeakerClassifier2(nn.Module):
    """
    A lightweight, CNN-based speaker classifier that focuses on local features.

    Args:
        input_dim (int): The dimension of the input embeddings (e.g., 256).
        num_channels (int): The number of channels in the CNN layers.
        output_dim (int): The dimension of the final speaker embedding (e.g., 256).
    """

    def __init__(self, input_dim=256, num_channels=256, output_dim=256, lambda_reversal=2.0):
        super(CnnSpeakerClassifier2, self).__init__()
        self.grad_revers = GradientReversal(lambda_reversal=lambda_reversal)
        self.cnn = nn.Sequential(weight_norm(nn.Conv1d(in_channels=input_dim, out_channels=num_channels, kernel_size=5, padding=2)), nn.ReLU(), weight_norm(nn.Conv1d(in_channels=num_channels, out_channels=num_channels, kernel_size=5, padding=2)), nn.ReLU(), weight_norm(nn.Conv1d(in_channels=num_channels, out_channels=output_dim, kernel_size=5, padding=2)))

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The input tensor of shape [B, T, D].
        """
        x = self.grad_revers(x)
        x = x.transpose(1, 2)
        outputs = self.cnn(x)
        outputs = torch.mean(outputs, dim=-1)
        return outputs

class MelVaeEncoder(nn.Module):
    """Posterior Encoder: Takes target mel and encodes it to a latent z."""

    def __init__(self, mel_bins: int=128, latent_dim: int=192, cnn_channels: int=512, rnn_hidden_dim: int=256, speaker_dim=192, useCausual=False, conv_kernel_size=5, layers=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.cnn = nn.Sequential(nn.Conv1d(mel_bins, latent_dim, kernel_size=1, padding='same'))
        self.wn_projection = WN(latent_dim, conv_kernel_size, 1, layers, gin_channels=speaker_dim)
        self.proj = nn.Conv1d(latent_dim, latent_dim * 2, 1)

    def forward(self, mel: torch.Tensor, g=torch.Tensor):
        x = mel.transpose(1, 2)
        g = g.transpose(1, 2)
        x = self.cnn(x)
        x = self.wn_projection(x, g)
        x = self.proj(x)
        (mu, log) = torch.split(x, self.latent_dim, dim=1)
        return (mu.transpose(1, 2), log.transpose(1, 2))

class ContentEncoder(nn.Module):

    def __init__(self, input_dim=2304, content_dim=192, speaker_dim=192, lambda_reversal=1.0, kernel_size=5, num_heads=4, window=16, dropout=0.1, n_blocks=2):
        super().__init__()
        self.pre = nn.Conv1d(input_dim, content_dim, kernel_size=5, padding=2)
        self.encoder_blocks = nn.Sequential(*[ConformerBlock3(d_model=content_dim, n_heads=num_heads, conv_kernel=5, ffn_mult=4, dropout=dropout, local_window=16 * 2 ** (n_blocks - i), d_cond=0, causal=False, rope=None) for i in range(n_blocks)])
        self.mu_proj = weight_norm(nn.Linear(content_dim, content_dim))
        self.logvar_proj = nn.Linear(content_dim, content_dim)
        self.classifer = CnnSpeakerClassifier2(content_dim, speaker_dim, speaker_dim, lambda_reversal=lambda_reversal)

    def forward(self, contentvec_feats, whisper_feats, hubert, f0, infer=False):
        input_feats = torch.cat([whisper_feats, hubert, contentvec_feats], dim=-1)
        latent_proj = self.pre(input_feats.transpose(1, 2)).transpose(1, 2)
        encodeded = self.encoder_blocks(latent_proj)
        mu = self.mu_proj(encodeded)
        logvar = self.logvar_proj(encodeded)
        std = torch.exp(logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        if infer:
            spk_pred = None
        else:
            spk_pred = self.classifer(encodeded)
        return (z, mu, logvar, spk_pred, None)

class FormantShiftHeadSimple(nn.Module):
    """
    ContentVec + F0 -> per-frame formant shift, zero-centered per sample.
    No mask, no bounding (output can take any real value).

    Inputs:
      content: (B, T, D_c)
      f0_hz:   (B, T) or (B, T, 1)

    Output:
      shift:   (B, T, 1), zero-mean per sample
    """

    def __init__(self, d_content: int, hidden: int=384, fix_rms: float | None=None, eps: float=1e-06, apply_norm=True):
        super().__init__()
        self.fix_rms = fix_rms
        self.eps = eps
        in_dim = d_content + 2
        if apply_norm:
            self.mlp = nn.Sequential(weight_norm(nn.Linear(in_dim, hidden, bias=True)), nn.SiLU(), weight_norm(nn.Linear(hidden, hidden, bias=True)), nn.SiLU(), weight_norm(nn.Linear(hidden, 1, bias=False)))
        else:
            self.mlp = nn.Sequential(nn.Linear(in_dim, hidden, bias=True), nn.SiLU(), nn.Linear(hidden, hidden, bias=True), nn.SiLU(), nn.Linear(hidden, 1, bias=False))

    def zscore_global(self, x: torch.Tensor, eps=1e-06):
        m = x.mean(dim=(0, 1), keepdim=True)
        s = x.std(dim=(0, 1), keepdim=True).clamp_min(eps)
        return (x - m) / s

    def forward(self, content, f0_hz: torch.Tensor):
        if f0_hz.dim() == 3:
            f0 = f0_hz.squeeze(-1)
        else:
            f0 = f0_hz
        f0_safe = f0.clamp_min(1.0)
        logf0 = torch.log(f0_safe)
        dlogf0 = F.pad(logf0[:, 1:] - logf0[:, :-1], (1, 0))
        logf0_n = self.zscore_global(logf0)[..., None]
        dlogf0_n = self.zscore_global(dlogf0)[..., None]
        x = torch.cat([content, logf0_n, dlogf0_n], dim=-1)
        y = self.mlp(x)
        y = y - y.mean(dim=1, keepdim=True)
        if self.fix_rms is not None:
            rms = torch.sqrt((y * y).mean(dim=1, keepdim=True) + self.eps)
            y = y * (self.fix_rms / (rms + self.eps))
        return y

    def init_zero(self):
        nn.init.zeros_(self.mlp[-1].weight)

class QExitAdaptiveDecoder(nn.Module):
    """Utterance-level Q-Exit controller around one weight-shared block.

    Q-Exit uses p_t = survival_t * sigmoid(gate_t) and exits at the first
    CDF >= q_exit. The last allowed step absorbs all remaining probability.
    Training evaluates every depth up to TRAIN_LOOP_LIMIT, so reconstruction and
    entropy losses can train the discrete gate without sampling. The main
    output is the same hard quantile-selected state used at inference; gate
    gradients come from the auxiliary expected reconstruction objective.

    A small factorized ContentVec head estimates reconstruction quality.
    Source ContentVec is available in both training and inference. Its
    detached error, improvement, hidden state and update inform the gate.
    Sinusoidal depth signals distinguish repeated applications of the block.
    Speaker conditioning is passed to the shared block at every depth.
    At inference, completed utterances and their speaker conditions are
    removed from subsequent loops together.
    Training checkpoints the shared block/head to reduce activation memory.

    Based on Q-Exit: https://tinkoff-ai.github.io/palbert/
    """
    TRAIN_LOOP_LIMIT = 20

    def __init__(self, d_model, content_dim=768, reconstruction_dim=64, max_loops=12, q_exit=0.5, gradient_checkpointing=True):
        super().__init__()
        for (name, value) in (('d_model', d_model), ('content_dim', content_dim), ('reconstruction_dim', reconstruction_dim), ('max_loops', max_loops)):
            if type(value) is not int or value <= 0:
                raise ValueError(f'{name} must be a positive integer')
        if not math.isfinite(q_exit) or not 0.0 <= q_exit <= 1.0:
            raise ValueError('q_exit must be finite and in [0, 1]')
        self.d_model = d_model
        self.content_dim = content_dim
        self.max_loops = max_loops
        self.q_exit = float(q_exit)
        self.gradient_checkpointing = gradient_checkpointing
        self.content_reconstruction = nn.Sequential(nn.Linear(d_model, reconstruction_dim), nn.SiLU(), nn.Linear(reconstruction_dim, content_dim))
        self.exit_gate = nn.Sequential(nn.Linear(2 * d_model + 3, reconstruction_dim), nn.SiLU(), nn.Linear(reconstruction_dim, 1))
        nn.init.normal_(self.exit_gate[-1].weight, std=0.001)
        nn.init.zeros_(self.exit_gate[-1].bias)

    def _step(self, hidden, content, previous_error, block, step, budget, speaker_feat):
        frequencies = torch.exp(torch.arange(0, self.d_model, 2, device=hidden.device, dtype=torch.float32) * (-math.log(10000.0) / self.d_model))
        angles = frequencies * step
        signal = torch.stack((angles.sin(), angles.cos()), dim=-1).flatten()[:self.d_model]
        updated = block(hidden + 0.1 * signal.to(hidden.dtype), speaker_feat)
        prediction = self.content_reconstruction(updated)
        error = (prediction.float() - content.float()).abs().mean(dim=(1, 2))
        improvement = previous_error - error.detach() if step > 1 else torch.zeros_like(error)
        gate_features = torch.cat((updated.mean(dim=1), (updated - hidden).abs().mean(dim=1), error.detach().log1p().unsqueeze(1), improvement.unsqueeze(1), error.new_full((hidden.shape[0], 1), step / budget)), dim=-1)
        logits = self.exit_gate(gate_features.to(updated.dtype)).squeeze(-1).float()
        return (updated, error, logits)

    def forward(self, hidden, block, content, adaptive=False, max_loops=None, q_exit=None, speaker_feat=None):
        if hidden.ndim != 3 or hidden.shape[-1] != self.d_model:
            raise ValueError('hidden must have shape (B, T, d_model)')
        if hidden.shape[0] == 0 or hidden.shape[1] == 0:
            raise ValueError('hidden must contain at least one sample and frame')
        if content.shape != (*hidden.shape[:2], self.content_dim):
            raise ValueError('content must have shape (B, T, content_dim), aligned with hidden')
        budget = self.max_loops if max_loops is None else max_loops
        quantile = self.q_exit if q_exit is None else q_exit
        if type(budget) is not int or budget <= 0:
            raise ValueError('max_loops must be a positive integer')
        if not math.isfinite(quantile) or not 0.0 <= quantile <= 1.0:
            raise ValueError('q_exit must be finite and in [0, 1]')
        if not adaptive or self.training:
            budget = min(budget, self.TRAIN_LOOP_LIMIT)
        content = content.detach()
        batch = hidden.shape[0]
        if speaker_feat is not None:
            if speaker_feat.ndim == 1:
                speaker_feat = speaker_feat.unsqueeze(0)
            if speaker_feat.ndim != 2 or speaker_feat.shape[0] not in (1, batch):
                raise ValueError('speaker_feat must have shape (B, speaker_dim) or (1, speaker_dim)')
            speaker_feat = speaker_feat.expand(batch, -1)
        active = torch.arange(batch, device=hidden.device)
        remaining = torch.ones(batch, device=hidden.device, dtype=torch.float32)
        previous_error = torch.zeros_like(remaining)
        output = torch.zeros_like(hidden)
        exit_steps = torch.zeros(batch, device=hidden.device, dtype=torch.long)
        full_cdf = torch.zeros_like(remaining)
        (probabilities, cdfs, errors) = ([], [], [])
        for step in range(1, budget + 1):
            if self.gradient_checkpointing and (not adaptive) and torch.is_grad_enabled():
                (updated, error, logits) = checkpoint(self._step, hidden, content, previous_error, block, step, budget, speaker_feat, use_reentrant=False)
            else:
                (updated, error, logits) = self._step(hidden, content, previous_error, block, step, budget, speaker_feat)
            probability = remaining if step == budget else remaining * torch.sigmoid(logits - math.log(budget - step))
            remaining = (remaining - probability).clamp_min(0.0)
            cdf = 1.0 - remaining
            stop = torch.ones_like(cdf, dtype=torch.bool) if step == budget else cdf >= quantile if quantile < 1.0 else torch.zeros_like(cdf, dtype=torch.bool)
            if adaptive:
                probabilities.append(probability.new_zeros(batch).index_copy(0, active, probability))
                full_cdf = full_cdf.index_copy(0, active, cdf)
                cdfs.append(full_cdf)
                finished = active[stop]
                output = output.index_copy(0, finished, updated[stop])
                exit_steps = exit_steps.index_fill(0, finished, step)
                if bool(stop.all()):
                    print(f'Exit at step {step}')
                    break
                active = active[~stop]
                hidden = updated[~stop]
                content = content[~stop]
                if speaker_feat is not None:
                    speaker_feat = speaker_feat[~stop]
                previous_error = error.detach()[~stop]
                remaining = remaining[~stop]
            else:
                probabilities.append(probability)
                cdfs.append(cdf)
                errors.append(error)
                newly_finished = stop & (exit_steps == 0)
                output = torch.where(newly_finished[:, None, None], updated, output)
                exit_steps = torch.where(newly_finished, step, exit_steps)
                hidden = updated
                previous_error = error.detach()
        probabilities = torch.stack(probabilities, dim=1)
        depths = torch.arange(1, probabilities.shape[1] + 1, device=hidden.device, dtype=torch.float32)
        expected_steps = (probabilities * depths).sum(dim=1) if not adaptive else exit_steps.float()
        entropy = -(probabilities * probabilities.clamp_min(1e-08).log()).sum(dim=1).mean()
        reconstruction_loss = probabilities.new_zeros(())
        if errors:
            errors = torch.stack(errors, dim=1)
            reconstruction_loss = (probabilities * errors).sum(dim=1).mean() + 0.1 * errors.mean()
        return (output, {'reconstruction_loss': reconstruction_loss, 'entropy': entropy, 'expected_steps': expected_steps, 'exit_steps': exit_steps, 'exit_probabilities': probabilities, 'exit_cdf': torch.stack(cdfs, dim=1), 'loops_run': step, 'loop_budget': budget})

class SoftVcStylizer(nn.Module):
    """ Multi-source input inspired by https://github.com/ShadowLoveElysia/Whisper-vits-svc-LargeV3"""
    def __init__(self, input_dim=2304, latent_dim=256, mel_bins=128, decoder_dim=768, num_flow_layers=4, decoder_layers=1, spk_dim=192, train_formant=True, decoder_max_loops=25, decoder_q_exit=0.75, decoder_reconstruction_dim=64, decoder_checkpoint=True):
        super().__init__()
        self.latent_dim = latent_dim
        self.spk_dim = spk_dim
        self.prior_encoder = ContentEncoder(input_dim=input_dim, content_dim=latent_dim, speaker_dim=spk_dim, n_blocks=3)
        self.posterior_encoder = MelVaeEncoder(mel_bins=mel_bins, latent_dim=latent_dim, useCausual=False, conv_kernel_size=5, speaker_dim=spk_dim, layers=4)
        self.train_formant = train_formant
        self.flow = ResidualCouplingBlockVits(latent_dim, latent_dim, 5, 1, num_flow_layers, gin_channels=spk_dim, n_flows=4)
        self.refiner = StylizerRefiner(dim=decoder_dim)
        self.decoder_input_proj = weight_norm(nn.Linear(latent_dim, decoder_dim))
        self.rope = None
        self.formant_estimator = FormantShiftHeadSimple(decoder_dim, apply_norm=False)
        self.decoder_blocks = nn.ModuleList([RoformerBlock(d_model=decoder_dim, n_heads=8, ffn_mult=1, dropout=0.1, d_cond=spk_dim, causal=False, rope=self.rope, local_window=64, n_layers=25)])
        self.decoder_exit = QExitAdaptiveDecoder(decoder_dim, content_dim=768, reconstruction_dim=decoder_reconstruction_dim, max_loops=decoder_max_loops, q_exit=decoder_q_exit, gradient_checkpointing=decoder_checkpoint)
        self.out_attn = ConformerBlock3(d_model=decoder_dim, n_heads=4, conv_kernel=15, ffn_mult=4, dropout=0.1, local_window=16, d_cond=0, causal=False, rope=None)
        self.output_proj = weight_norm(nn.Linear(decoder_dim, decoder_dim))

    def forward(self, contentvec_feats, whisper_feats, hubert_feats, speaker_feat, mel, f0, vol, steps=0, infer=False, noise_fac=0.0001, t_start=1.0, infer_step=50, alpha=0.0, latent_only=False, return_decoder_aux=False, max_decoder_loops=None, q_exit=None):
        (B, L, _) = contentvec_feats.shape
        speaker_feat_orig = speaker_feat
        if len(speaker_feat_orig.shape) == 1:
            speaker_feat_orig = speaker_feat_orig.unsqueeze(0)
        speaker_feat_orig = F.normalize(speaker_feat_orig, dim=1)
        if infer:
            contentvec_feats_proj = contentvec_feats
            whisper_feats += torch.randn_like(whisper_feats) * noise_fac
            hubert_feats += torch.randn_like(hubert_feats) * 0
            if len(speaker_feat.shape) == 1:
                speaker_feat = speaker_feat.unsqueeze(0)
        else:
            contentvec_feats_proj = contentvec_feats + torch.randn_like(contentvec_feats) * 2
            whisper_feats += torch.randn_like(whisper_feats) * 1.0
            hubert_feats += torch.randn_like(hubert_feats) * 2.0
        if not infer:
            if len(speaker_feat.shape) == 1:
                speaker_feat = F.normalize(speaker_feat).unsqueeze(0).unsqueeze(0).expand(-1, L, -1)
            else:
                speaker_feat = F.normalize(speaker_feat).unsqueeze(1).expand(-1, L, -1)
            (z_pr, mu_pr, logvar_pr, spk_pred, mask) = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats, f0, infer=False)
            (mu_ps, logvar_ps) = self.posterior_encoder(mel, speaker_feat)
            z_ps = mu_ps + torch.randn_like(mu_ps) * torch.exp(logvar_ps)
            (z_fwd, logdet_fwd) = self.flow(z_ps.transpose(1, 2), g=speaker_feat.transpose(1, 2))
            (z_bkw, log_det_bkw) = self.flow(z_pr.transpose(1, 2), g=speaker_feat.transpose(1, 2), reverse=True)
            z_fwd = z_fwd.transpose(1, 2)
            z_bkw = z_bkw.transpose(1, 2)
        else:
            speaker_feat = F.normalize(speaker_feat).unsqueeze(1).expand(-1, L, -1)
            (z_pr, mu_pr, logvar_pr, spk_pred, mask) = self.prior_encoder(contentvec_feats_proj, whisper_feats, hubert_feats, f0, infer=True)
            (z_bkw, log_det_bkw) = self.flow(z_pr.transpose(1, 2), g=speaker_feat.transpose(1, 2), reverse=True)
            z_bkw = z_bkw.transpose(1, 2)
            mu_ps = None
            logvar_ps = None
            z_ps = None
            logdet_fwd = None
            spk_pred = None
            z_fwd = None
        if infer:
            x = z_bkw
        elif steps >= 30000:
            p = torch.rand(1)
            if p.item() >= 0.3:
                x = z_ps
            else:
                x = z_bkw
        else:
            x = z_ps
        x = self.decoder_input_proj(x)
        (x, decoder_aux) = self.decoder_exit(x, self.decoder_blocks[0], contentvec_feats, adaptive=infer, max_loops=max_decoder_loops if not infer else 20, q_exit=q_exit if not infer else 0.75, speaker_feat=speaker_feat_orig)
        x = self.out_attn(x)
        decode_proj = self.output_proj(x)
        stylized_feats = self.refiner(decode_proj)
        if self.train_formant:
            formant = self.formant_estimator(stylized_feats, f0)
        else:
            formant = torch.tensor(0)
        outputs = (stylized_feats, mu_pr, logvar_pr, mu_ps, logvar_ps, z_fwd, z_pr, z_bkw, log_det_bkw, spk_pred, formant, logdet_fwd, z_ps, mask)
        return outputs + (decoder_aux,) if return_decoder_aux else outputs
