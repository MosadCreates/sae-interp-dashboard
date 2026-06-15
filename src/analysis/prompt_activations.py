"""Compute feature activations for arbitrary prompts.

Given a text prompt, runs GPT-2 + SAE and returns the feature
activation matrix for every token position.
"""

from __future__ import annotations

from typing import List, Tuple

import torch

from src.model import ActivationExtractor
from src.sae.model import SparseAutoencoder


@torch.no_grad()
def get_prompt_activations(
    prompt: str,
    extractor: ActivationExtractor,
    sae: SparseAutoencoder,
) -> dict:
    """Analyse a prompt and return per-token feature activations.

    Args:
        prompt: Input text string.
        extractor: ActivationExtractor with loaded GPT-2 model.
        sae: Trained SparseAutoencoder.

    Returns:
        dict with:
            tokens: List of str, tokenised prompt.
            token_ids: List of int, token IDs.
            feature_activations: List of List[dict], per-token top feature activations
                where each entry is {"feature_id": int, "value": float}.
            top_features: List of dict, global top-k features across the prompt
                with {"feature_id": int, "max_activation": float}.
            full_matrix: List of List[float], [seq_len, d_sae] activation matrix
                (only includes features that are active anywhere in the prompt).
    """
    device = next(sae.parameters()).device
    top_k_features = 10

    # Tokenise
    enc = extractor.tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    tokens = extractor.tokenizer.convert_ids_to_tokens(input_ids[0].tolist())

    # Get activations
    acts = extractor(input_ids, enc.get("attention_mask"))  # [1, seq_len, d_model]
    acts = acts.to(device)

    # SAE encode
    seq_len = acts.size(1)
    acts_flat = acts.view(-1, acts.size(-1))  # [seq_len, d_model]
    feature_acts = sae.encode(acts_flat)  # [seq_len, d_sae]

    # Per-token top-K features
    per_token: List[List[dict]] = []
    full_matrix: List[List[float]] = []
    active_feature_ids: set = set()

    for pos in range(seq_len):
        f_acts = feature_acts[pos]  # [d_sae]
        nonzero_mask = f_acts > 0
        nonzero_indices = torch.where(nonzero_mask)[0]
        nonzero_values = f_acts[nonzero_indices]

        # Sort by activation descending
        sorted_idx = nonzero_values.argsort(descending=True)
        top_n = min(top_k_features, len(sorted_idx))
        position_features = []
        for i in range(top_n):
            fid = nonzero_indices[sorted_idx[i]].item()
            val = nonzero_values[sorted_idx[i]].item()
            position_features.append({"feature_id": fid, "value": round(val, 4)})
            active_feature_ids.add(fid)

        per_token.append(position_features)

        # Build full matrix row (only for active features to limit size)
        row = []
        for fid in active_feature_ids:
            row.append(round(f_acts[fid].item(), 4))
        full_matrix.append(row)

    # Global top-K features across prompt
    global_vals: dict = {}
    for pos_features in per_token:
        for entry in pos_features:
            fid = entry["feature_id"]
            global_vals[fid] = max(global_vals.get(fid, 0.0), entry["value"])

    top_features = sorted(global_vals.items(), key=lambda x: -x[1])[:20]
    top_features_list = [
        {"feature_id": fid, "max_activation": round(val, 4)}
        for fid, val in top_features
    ]

    # Build active feature IDs list for matrix interpretation
    active_fids = sorted(active_feature_ids)

    return {
        "tokens": tokens,
        "token_ids": input_ids[0].tolist(),
        "per_token_features": per_token,
        "top_features": top_features_list,
        "full_matrix": full_matrix,
        "active_feature_ids": active_fids,
    }
