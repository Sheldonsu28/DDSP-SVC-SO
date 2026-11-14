"""Selected dependencies of the SoftVcStylizer frontend workflow."""

import torch

def map_normal_diagonal(x, mu_src, cov_src, mu_tgt, cov_tgt):
    """
    Maps tensor x (N, T, D) from Source Dist to Target Dist.
    Assumes Diagonal Covariance (independent features).
    
    Args:
        x (Tensor): Input tensor of shape [1, T, dim]
        mu_src (Tensor): Source mean [1, dim]
        cov_src (Tensor): Source standard deviation [1, dim] (legacy argument name).
        mu_tgt (Tensor): Target mean [1, dim]
        cov_tgt (Tensor): Target standard deviation [1, dim] (legacy argument name).
    """
    sigma_src = cov_src
    sigma_tgt = cov_tgt
    z = (x - mu_src) / sigma_src
    y = z * sigma_tgt + mu_tgt
    return y
