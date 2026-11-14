"""Selected dependencies of the SoftVcStylizer frontend workflow."""

import math
import torch
from torch import nn
import torch.nn.functional as F

class FiLM(nn.Module):
    """Feature-wise Linear Modulation conditioned on speaker embedding."""

    def __init__(self, d_cond, d_model):
        super().__init__()
        self.to_gamma_beta = nn.Linear(d_cond, 2 * d_model)
        nn.init.zeros_(self.to_gamma_beta.weight)
        nn.init.zeros_(self.to_gamma_beta.bias)

    def forward(self, x, cond):
        gb = self.to_gamma_beta(cond)
        (gamma, beta) = gb.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(1) + 1.0
        beta = beta.unsqueeze(1)
        return gamma * x + beta

class SwiGLU(nn.Module):
    """
    SwiGLU feed-forward (LLaMA-style): down(SiLU(gate(x)) * up(x)).

    The gate and up projections share a single fused input Linear. All
    projections are bias-free. Dropout is applied to the gated hidden state
    and to the output, matching the two dropout sites of the MLP it replaces.
    """

    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.w_in = nn.Linear(d_model, 2 * d_ff, bias=False)
        self.w_out = nn.Linear(d_ff, d_model, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        (gate, up) = self.w_in(x).chunk(2, dim=-1)
        h = self.drop(F.silu(gate) * up)
        return self.drop(self.w_out(h))

class ConformerBlock3(nn.Module):
    """RMSNorm Conformer with correct SDPA masks and optional causal operation.

    Keeps the ConformerBlock2 constructor and forward interfaces, macaron
    SwiGLU FFNs, RoPE support, residual initialization, and optional FiLM.
    All normalization is per frame, including inside the convolution module;
    no normalization statistics are shared across time. With causal=True,
    both attention and convolution use only the current and earlier frames.
    This is a full-sequence block; it does not maintain a streaming KV/conv cache.

    local_window > 0 is an attention radius (left-only when causal), while
    local_window <= 0 enables full attention, still respecting causal=True.
    RMSNorm and the convolution norm differ from ConformerBlock2, so its
    checkpoints are not directly interchangeable with this class.
    """

    def __init__(self, d_model=512, n_heads=8, ffn_mult=2, conv_kernel=15, dropout=0.1, d_cond=512, local_window=12, causal=False, rope=None, n_layers=1):
        super().__init__()
        if d_model <= 0 or n_heads <= 0 or d_model % n_heads != 0:
            raise ValueError('d_model must be positive and divisible by n_heads')
        if conv_kernel <= 0 or conv_kernel % 2 == 0:
            raise ValueError('conv_kernel must be a positive odd integer')
        if d_cond < 0:
            raise ValueError('d_cond must be non-negative')
        d_ff = d_model * ffn_mult
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.local_window = local_window
        self.causal = causal
        self.rope = rope
        self.norm_ff1 = nn.RMSNorm(d_model, eps=1e-06)
        self.ff1 = SwiGLU(d_model, d_ff, dropout)
        self.norm_ff2 = nn.RMSNorm(d_model, eps=1e-06)
        self.ff2 = SwiGLU(d_model, d_ff, dropout)
        self.norm_attn = nn.RMSNorm(d_model, eps=1e-06)
        self.attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop_p = dropout
        self.norm_conv = nn.RMSNorm(d_model, eps=1e-06)
        self.conv_pw1 = nn.Conv1d(d_model, 2 * d_model, 1, bias=False)
        self.conv_dw = nn.Conv1d(d_model, d_model, conv_kernel, padding=0, groups=d_model, bias=False)
        self.conv_norm = nn.RMSNorm(d_model, eps=1e-06)
        self.conv_pw2 = nn.Conv1d(d_model, d_model, 1, bias=False)
        self.conv_drop = nn.Dropout(dropout)
        self.norm_final = nn.RMSNorm(d_model, eps=1e-06)
        self.film = FiLM(d_cond, d_model) if d_cond > 0 else None
        self._init_residual_scale(n_layers)
        self.register_buffer('_mask_cache', None, persistent=False)
        self._mask_key = None

    def _make_local_mask(self, seq_len, window_size, device, causal=False):
        key = (seq_len, window_size, causal, device)
        if self._mask_key == key:
            return self._mask_cache
        idx = torch.arange(seq_len, device=device)
        diff = idx[None, :] - idx[:, None]
        if causal:
            allowed = (diff <= 0) & (diff >= -window_size)
        else:
            allowed = diff.abs() <= window_size
        self._mask_cache = allowed
        self._mask_key = key
        return allowed

    @torch.no_grad()
    def _init_residual_scale(self, n_layers):
        scale = 1.0 / math.sqrt(2.0 * max(1, n_layers))
        for module in (self.attn_out, self.ff1.w_out, self.ff2.w_out, self.conv_pw2):
            module.weight.mul_(scale)

    def forward(self, x, cond=None):
        x = x + 0.5 * self.ff1(self.norm_ff1(x))
        h = self.norm_attn(x)
        (B, T, _) = h.shape
        qkv = self.attn_qkv(h).view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        (q, k, v) = (qkv[0], qkv[1], qkv[2])
        if self.rope is not None:
            (q, k) = self.rope(q, k)
        if self.local_window > 0:
            attn_mask = self._make_local_mask(T, self.local_window, x.device, causal=self.causal)
            is_causal = False
        else:
            attn_mask = None
            is_causal = self.causal
        h = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=self.attn_drop_p if self.training else 0.0, is_causal=is_causal)
        h = h.transpose(1, 2).contiguous().view(B, T, -1)
        x = x + self.attn_out(h)
        h = self.norm_conv(x).transpose(1, 2)
        h = F.glu(self.conv_pw1(h), dim=1)
        kernel = self.conv_dw.kernel_size[0]
        padding = (kernel - 1, 0) if self.causal else (kernel // 2, kernel // 2)
        h = self.conv_dw(F.pad(h, padding))
        h = self.conv_norm(h.transpose(1, 2)).transpose(1, 2)
        h = self.conv_pw2(F.silu(h))
        x = x + self.conv_drop(h).transpose(1, 2)
        x = x + 0.5 * self.ff2(self.norm_ff2(x))
        x = self.norm_final(x)
        if self.film is not None and cond is not None:
            x = self.film(x, cond)
        return x

class RoformerBlock(nn.Module):
    """
    GPT/LLaMA-style RoFormer encoder layer: a pre-norm Transformer block
    (MHSA + FFN) whose only source of position information is RoPE applied to
    q/k inside attention. Compared to `ConformerBlock2` it drops the
    convolution module and the macaron half-step FFNs, so it is lighter and
    supports either global or local attention without convolution.

    Shares the same modern decoder-block internals as `ConformerBlock3`:
    RMSNorm (pre-norm), a SwiGLU feed-forward, bias-free projections, and
    GPT-2 scaled residual init (1/sqrt(2 * n_layers)). The constructor
    keeps the existing RoPE / local-window / causal / FiLM interface.
    FiLM is optional when cond is None. local_window <= 0 means full
    attention, still respecting causal=True. Boolean SDPA masks use
    True for allowed keys. RMSNorm replaces the former LayerNorms, so
    old LayerNorm checkpoints are not directly interchangeable.
    """

    def __init__(self, d_model=512, n_heads=8, ffn_mult=4, dropout=0.1, d_cond=512, local_window=0, causal=False, rope=None, n_layers=1):
        super().__init__()
        d_ff = d_model * ffn_mult
        self.norm_attn = nn.RMSNorm(d_model, eps=1e-06)
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert d_model % n_heads == 0, 'd_model must be divisible by n_heads'
        self.attn_qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.attn_out = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop_p = dropout
        self.rope = rope
        self.norm_ffn = nn.RMSNorm(d_model, eps=1e-06)
        self.ffn = SwiGLU(d_model, d_ff, dropout)
        self.local_window = local_window
        self.causal = causal
        self.d_cond = d_cond
        self.norm_final = nn.RMSNorm(d_model, eps=1e-06)
        if self.d_cond > 0:
            self.film = FiLM(d_cond, d_model)
        else:
            self.film = None
        self._init_residual_scale(n_layers)
        self.register_buffer('_mask_cache', None, persistent=False)
        self._mask_key = None

    def _make_local_mask(self, seq_len, window_size, device, causal=False):
        key = (seq_len, window_size, causal, device)
        if self._mask_key == key:
            return self._mask_cache
        idx = torch.arange(seq_len, device=device)
        diff = idx[None, :] - idx[:, None]
        if causal:
            allowed = (diff <= 0) & (diff >= -window_size)
        else:
            allowed = diff.abs() <= window_size
        mask = allowed
        self._mask_cache = mask
        self._mask_key = key
        return mask

    @torch.no_grad()
    def _init_residual_scale(self, n_layers):
        scale = 1.0 / math.sqrt(2.0 * max(1, n_layers))
        for module in (self.attn_out, self.ffn.w_out):
            module.weight.mul_(scale)

    def forward(self, x, cond=None):
        h = self.norm_attn(x)
        (B, T, _) = h.shape
        qkv = self.attn_qkv(h)
        qkv = qkv.view(B, T, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        (q, k, v) = (qkv[0], qkv[1], qkv[2])
        if self.rope is not None:
            (q, k) = self.rope(q, k)
        is_causal = False
        if self.causal and self.local_window > 0:
            attn_mask = self._make_local_mask(T, self.local_window, x.device, causal=True)
        elif self.causal:
            attn_mask = None
            is_causal = True
        elif self.local_window > 0:
            attn_mask = self._make_local_mask(T, self.local_window, x.device)
        else:
            attn_mask = None
        h = F.scaled_dot_product_attention(q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0, is_causal=is_causal, attn_mask=attn_mask)
        h = h.transpose(1, 2).contiguous().view(B, T, -1)
        h = self.attn_out(h)
        x = x + h
        x = x + self.ffn(self.norm_ffn(x))
        x = self.norm_final(x)
        if self.film is not None and cond is not None:
            x = self.film(x, cond)
        return x
