"""Loss functions and evaluation metrics for SAE training."""

from __future__ import annotations

import torch


def sae_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    acts: torch.Tensor,
    lambda_l1: float = 8e-5,
) -> dict[str, torch.Tensor]:
    """Compute the SAE loss and auxiliary metrics.

    Args:
        x:        Original input activations [batch, d_model].
        x_recon:  Reconstructed activations [batch, d_model].
        acts:     Feature activations [batch, d_sae].
        lambda_l1: L1 sparsity coefficient.

    Returns:
        dict with keys:
            total_loss:       l2_loss + l1_loss (scalar, for backprop).
            l2_loss:          Mean squared reconstruction error (scalar).
            l1_loss:          Lambda * mean L1 of feature activations (scalar).
            l0_norm:          Mean number of non-zero features per token (scalar).
            explained_variance: 1 - Var(residual) / Var(x) (scalar, higher = better).
            l0:               Same as l0_norm (convenience).
    """
    l2_loss = (x - x_recon).pow(2).mean()

    l1_loss = lambda_l1 * acts.float().norm(p=1, dim=-1).mean()

    total_loss = l2_loss + l1_loss

    with torch.no_grad():
        l0_norm = (acts > 0).float().sum(dim=-1).mean()

        residual_var = (x - x_recon).var(dim=-1).mean()
        x_var = x.var(dim=-1).mean() + 1e-8
        explained_variance = 1.0 - (residual_var / x_var)

    return {
        "total_loss": total_loss,
        "l2_loss": l2_loss.detach(),
        "l1_loss": l1_loss.detach(),
        "l0_norm": l0_norm,
        "explained_variance": explained_variance,
    }
