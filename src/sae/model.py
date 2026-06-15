"""Sparse Autoencoder — core architecture."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """Sparse Autoencoder trained on transformer activations.

    Architecture:
        x_centered = x - b_pre
        pre_acts   = x_centered @ W_enc + b_enc
        acts       = ReLU(pre_acts)
        x_recon    = acts @ W_dec + b_pre

    The decoder columns (W_dec rows) are constrained to unit norm.
    W_enc is NOT tied to W_dec (unlike tied autoencoders).
    """

    def __init__(
        self,
        d_model: int = 768,
        d_sae: int = 6144,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_sae = d_sae

        # --- Encoder ---
        self.W_enc = nn.Parameter(torch.empty(d_model, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # --- Pre-encoder bias (centers input) ---
        self.b_pre = nn.Parameter(torch.zeros(d_model))

        # --- Decoder ---
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_model))

        self._init_weights()

    def _init_weights(self) -> None:
        # W_enc: init from a normal distribution
        nn.init.kaiming_uniform_(self.W_enc, a=0.0, mode="fan_in", nonlinearity="relu")

        # W_dec: init close to the transpose of W_enc (SAELens convention)
        # This gives the decoder a head start aligned with the encoder
        with torch.no_grad():
            w_enc_T = self.W_enc.data.T.clone()  # [d_sae, d_model]
            # Normalise each row of W_dec to unit norm
            w_enc_T = w_enc_T / (w_enc_T.norm(dim=1, keepdim=True) + 1e-8)
            self.W_dec.data.copy_(w_enc_T)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input activations [batch, d_model] or [batch, seq_len, d_model].

        Returns:
            acts:    Feature activations [batch, d_sae] or [batch, seq_len, d_sae].
            x_recon: Reconstructed activations, same shape as x.
        """
        x_centered = x - self.b_pre  # centre input

        pre_acts = x_centered @ self.W_enc + self.b_enc  # [..., d_sae]
        acts = F.relu(pre_acts)

        x_recon = acts @ self.W_dec + self.b_pre  # [..., d_model]

        return acts, x_recon

    @torch.no_grad()
    def normalize_decoder_(self) -> None:
        """Project each decoder row to unit Euclidean norm.

        Must be called after every gradient step during training.
        """
        norms = self.W_dec.norm(dim=1, keepdim=True)
        self.W_dec.data.div_(norms.clamp_min_(1e-8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode input to feature activations (without reconstruction)."""
        x_centered = x - self.b_pre
        return F.relu(x_centered @ self.W_enc + self.b_enc)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        """Decode feature activations back to activation space."""
        return acts @ self.W_dec + self.b_pre
