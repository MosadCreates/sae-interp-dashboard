"""Dead feature detection and neuron resampling.

Implements the resampling technique from Anthropic's SAE work:
  - Track which features fire (activation > 0) over a sliding window.
  - Periodically identify dead features (zero activations in the window).
  - Resample dead features from high-loss training examples.
"""

from __future__ import annotations

import torch


class DeadFeatureTracker:
    """Tracks recently active features and identifies dead features.

    Maintains a boolean buffer of size n_dead_steps x d_sae.
    A feature is "alive" if it fired at least once in the last n_dead_steps steps.
    """

    def __init__(self, d_sae: int, n_dead_steps: int = 12_500):
        self.d_sae = d_sae
        self.n_dead_steps = n_dead_steps
        self.buffer: list[torch.Tensor] = []
        self.steps_since_reset = 0

    def update(self, acts: torch.Tensor) -> None:
        """Record which features fired in the current batch.

        Args:
            acts: Feature activations [batch, d_sae].
        """
        fired = (acts > 0).any(dim=0).detach().cpu()  # [d_sae]
        self.buffer.append(fired)
        self.steps_since_reset += 1

        if len(self.buffer) > self.n_dead_steps:
            self.buffer.pop(0)

    def get_dead_mask(self) -> torch.Tensor:
        """Return a boolean mask of dead features.

        A feature is dead if it never fired in the buffer.
        """
        if not self.buffer:
            return torch.zeros(self.d_sae, dtype=torch.bool)

        alive = torch.stack(self.buffer, dim=0).any(dim=0)
        return ~alive

    def num_dead(self) -> int:
        return self.get_dead_mask().sum().item()

    def reset(self) -> None:
        self.buffer = []
        self.steps_since_reset = 0


def resample_dead_features(
    sae: "SparseAutoencoder",
    dead_mask: torch.Tensor,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    n_dead_to_resample: int | None = None,
) -> None:
    """Resample dead feature encoder/decoder directions from high-loss examples.

    For each dead feature, we:
      1. Find the top-k highest-loss inputs from a batch of data.
      2. Re-initialise the encoder column to the normalised input vector.
      3. Re-initialise the decoder row to the same direction (unit-normed).
      4. Reset the encoder bias to zero.

    This is a simplified version of Anthropic's resampling procedure.
    """
    from src.sae.model import SparseAutoencoder

    dead_indices = torch.where(dead_mask)[0]  # [n_dead]

    if len(dead_indices) == 0:
        return

    if n_dead_to_resample is not None:
        perm = torch.randperm(len(dead_indices))[:n_dead_to_resample]
        dead_indices = dead_indices[perm]

    # Collect high-loss examples from one pass through the dataloader
    all_high_loss_inputs: list[torch.Tensor] = []
    all_high_loss_candidates: int = max(len(dead_indices) * 2, 256)

    sae.eval()
    with torch.no_grad():
        collected = 0
        for batch in dataloader:
            x = batch.to(device)
            acts, x_recon = sae(x)
            losses = (x - x_recon).pow(2).mean(dim=-1)  # [batch]
            _, idx = losses.topk(k=min(len(dead_indices) * 2, x.size(0)), dim=0)
            all_high_loss_inputs.append(x[idx].cpu())
            collected += len(idx)
            if collected >= all_high_loss_candidates:
                break

    if not all_high_loss_inputs:
        return

    high_loss_inputs = torch.cat(all_high_loss_inputs, dim=0)  # [N, d_model]

    n_to_resample = min(len(dead_indices), high_loss_inputs.size(0))
    dead_indices = dead_indices[:n_to_resample]
    resample_vecs = high_loss_inputs[:n_to_resample]  # [n, d_model]

    # Normalise the resample vectors
    resample_vecs = resample_vecs / (resample_vecs.norm(dim=1, keepdim=True) + 1e-8)

    with torch.no_grad():
        # Re-init encoder columns
        sae.W_enc.data[:, dead_indices] = resample_vecs.T.to(sae.W_enc.device)

        # Re-init decoder rows to the normalised encoder direction
        sae.W_dec.data[dead_indices] = resample_vecs.to(sae.W_dec.device)

        # Reset encoder biases to zero
        sae.b_enc.data[dead_indices] = 0.0

    sae.train()
