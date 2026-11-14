"""Selected dependencies of the SoftVcStylizer frontend workflow."""

import torch

def vits_kl_loss(z_p, logs_q, m_p, logs_p, total_logdet):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    total_logdet: [b] - total_logdet summed over each batch
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * (z_p - m_p) ** 2 * torch.exp(-2.0 * logs_p)
    return (torch.sum(kl) - torch.sum(total_logdet)) / z_p.numel()
