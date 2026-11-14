"""Selected dependencies of the SoftVcStylizer frontend workflow."""

import torch
from torch import nn
from .wavenet import WN

class ResidualCouplingLayer(nn.Module):

    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=0, gin_channels=0, mean_only=False):
        assert channels % 2 == 0, 'channels should be divisible by 2'
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.half_channels = channels // 2
        self.mean_only = mean_only
        self.pre = nn.Conv1d(self.half_channels, hidden_channels, 1)
        self.enc = WN(hidden_channels, kernel_size, dilation_rate, n_layers, p_dropout=p_dropout)
        self.post = nn.Conv1d(hidden_channels, self.half_channels * (2 - mean_only), 1)
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()
        self.snac = nn.Conv1d(gin_channels, 2 * self.half_channels, 1)

    def forward(self, x, g=None, reverse=False):
        speaker = self.snac(g)
        (speaker_m, speaker_v) = speaker.chunk(2, dim=1)
        (x0, x1) = torch.split(x, [self.half_channels] * 2, 1)
        x0_norm = (x0 - speaker_m) * torch.exp(-speaker_v)
        h = self.pre(x0_norm)
        h = self.enc(h)
        stats = self.post(h)
        if not self.mean_only:
            (m, logs) = torch.split(stats, [self.half_channels] * 2, 1)
        else:
            m = stats
            logs = torch.zeros_like(m)
        if not reverse:
            x1_norm = (x1 - speaker_m) * torch.exp(-speaker_v)
            x1 = m + x1_norm * torch.exp(logs)
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(logs, [1, 2]) - torch.sum(speaker_v.expand(-1, -1, logs.size(-1)), [1, 2])
            return (x, logdet)
        else:
            x1 = (x1 - m) * torch.exp(-logs)
            x1 = speaker_m + x1 * torch.exp(speaker_v)
            x = torch.cat([x0, x1], 1)
            logdet = torch.sum(-logs, [1, 2]) + torch.sum(speaker_v.expand(-1, -1, logs.size(-1)), [1, 2])
            return (x, logdet)

    def remove_weight_norm(self):
        self.enc.remove_weight_norm()

    def zero_last_layer(self):
        self.post.weight.data.zero_()
        self.post.bias.data.zero_()

class Flip(nn.Module):

    def forward(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        return (x, logdet)

    def inverse(self, x, *args, reverse=False, **kwargs):
        x = torch.flip(x, [1])
        logdet = torch.zeros(x.size(0)).to(dtype=x.dtype, device=x.device)
        return (x, logdet)

class ResidualCouplingBlockVits(nn.Module):

    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.flows = nn.ModuleList()
        self.n_flows = n_flows
        for i in range(n_flows):
            self.flows.append(ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
            self.flows.append(Flip())

    def forward(self, x, g=None, reverse=False):
        if not reverse:
            total_logdet = 0
            for flow in self.flows:
                (x, log_det) = flow(x, g=g, reverse=reverse)
                total_logdet += log_det
            return (x, total_logdet)
        else:
            total_logdet = 0
            for flow in reversed(self.flows):
                (x, log_det) = flow(x, g=g, reverse=reverse)
                total_logdet += log_det
            return (x, total_logdet)

    def remove_weight_norm(self):
        for i in range(self.n_flows):
            self.flows[i * 2].remove_weight_norm()

    def zero_last_layer(self):
        for i in range(self.n_flows):
            self.flows[i * 2].zero_last_layer()
