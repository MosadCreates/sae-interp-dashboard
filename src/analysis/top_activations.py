"""Find top-K activating contexts for every SAE feature.

Processes the corpus in a single pass through GPT-2 + SAE,
maintaining a running top-K heap per feature. Supports activation
frequency histograms and streaming processing.
"""

from __future__ import annotations

import heapq
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from tqdm import tqdm

from src.data.corpus import create_corpus_dataset
from src.model import ActivationExtractor
from src.sae.model import SparseAutoencoder


@torch.no_grad()
def compute_top_activations(
    sae: SparseAutoencoder,
    extractor: ActivationExtractor,
    config: dict,
    device: torch.device,
) -> dict:
    """Stream through the corpus and collect top-K contexts per feature.

    Returns a dict::
        {
          "features": {
            feature_id: {
              "top_contexts": [
                {"tokens": [...], "activations": [...], "max_activation": float},
                ...
              ],
              "max_activation": float,
              "mean_activation": float,
              "frequency": float,
              "histogram": [{"bin_center": float, "count": int}, ...],
            },
            ...
          },
          "metadata": {...}
        }
    """
    d_sae = sae.d_sae
    top_k = config.get("n_top_tokens", 50)
    context_window = config.get("context_window", 10)
    n_histogram_bins = config.get("n_histogram_bins", 100)
    max_tokens = config.get("max_analysis_tokens", 5_000_000)
    batch_size = config.get("batch_size", 8)

    corpus = create_corpus_dataset(
        tokenizer=extractor.tokenizer,
        context_length=extractor.tokenizer.model_max_length or 128,
        max_tokens=max_tokens,
        corpus_name=config.get("corpus_name", "roneneldan/TinyStories"),
    )
    dataloader = torch.utils.data.DataLoader(
        corpus, batch_size=batch_size, num_workers=config.get("num_workers", 0)
    )

    # Per-feature top-K heaps: list of (neg_activation, entry_id, entry_dict)
    # We store negative activations to get max-heap behaviour from heapq (min-heap).
    feature_heaps: List[List[Tuple[float, int, dict]]] = [
        [] for _ in range(d_sae)
    ]
    feature_histograms: List[List[int]] = [
        [0] * n_histogram_bins for _ in range(d_sae)
    ]
    feature_fire_counts: List[int] = [0] * d_sae
    total_tokens = 0
    entry_counter = 0

    # Histogram range: we'll track activations from 0 to max seen
    global_max_act = 1.0

    pbar = tqdm(desc="Analysing features", unit="tok")
    for batch_tokens in dataloader:
        token_ids = batch_tokens.to(device)  # [batch, seq_len]

        # Get residual stream activations
        acts = extractor(token_ids)  # [batch, seq_len, d_model]
        b, s, d = acts.shape
        acts_flat = acts.reshape(-1, d).to(device)  # [batch*seq, d_model]

        # SAE encode
        feature_acts = sae.encode(acts_flat)  # [batch*seq, d_sae]

        # Decode tokens for context
        tokens_2d = token_ids.cpu().tolist()  # list of lists

        # Update tracking for each position
        for pos in range(b * s):
            feat_acts = feature_acts[pos].cpu().numpy()  # [d_sae]
            active_indices = np.where(feat_acts > 0)[0]

            # Build context window (±context_window tokens around this position)
            batch_idx = pos // s
            seq_idx = pos % s
            ctx_start = max(0, seq_idx - context_window)
            ctx_end = min(s, seq_idx + context_window + 1)
            ctx_token_ids = token_ids[batch_idx, ctx_start:ctx_end].cpu().tolist()
            ctx_tokens = extractor.tokenizer.convert_ids_to_tokens(ctx_token_ids)

            # For context activations, we need per-token feature activations
            # We only store the activation of the current feature being tracked
            # The "context_acts" here stores all features' activations for display
            # Actually for the entry, we just need the main feature's activation
            # at the center token. The context tokens show surrounding context
            # without per-token activation values (to keep data small).

            for fid in active_indices:
                val = float(feat_acts[fid])
                feature_fire_counts[fid] += 1
                global_max_act = max(global_max_act, val)

                # Update histogram
                bin_idx = min(n_histogram_bins - 1, int(val))
                feature_histograms[fid][bin_idx] += 1

                # Update top-K heap (only if val is large enough)
                heap = feature_heaps[fid]
                entry = (
                    -val,  # negate for max-heap
                    entry_counter,
                    {
                        "tokens": ctx_tokens,
                        "activation": val,
                    },
                )
                if len(heap) < top_k:
                    heapq.heappush(heap, entry)
                elif -entry[0] > -heap[0][0]:  # val > smallest in heap
                    heapq.heappushpop(heap, entry)
                entry_counter += 1

            total_tokens += 1

        pbar.update(b * s)
        if total_tokens >= max_tokens:
            break

    pbar.close()

    # --- Build output ---
    features_out: Dict[int, Dict[str, Any]] = {}

    for fid in range(d_sae):
        heap = feature_heaps[fid]
        # Sort by activation descending
        sorted_entries = sorted(heap, key=lambda x: -x[0])  # most negative first = highest activation

        top_contexts = []
        max_act = 0.0
        sum_act = 0.0
        for neg_val, _, entry in sorted_entries:
            val = -neg_val
            max_act = max(max_act, val)
            sum_act += val
            top_contexts.append(entry)

        n_active = feature_fire_counts[fid]
        frequency = n_active / max(1, total_tokens)

        # Build histogram
        hist_bins = []
        nonzero_indices = np.where(np.array(feature_histograms[fid]) > 0)[0]
        if len(nonzero_indices) > 0:
            hist_max = max(nonzero_indices) + 1
        else:
            hist_max = 1
        for bin_i in range(hist_max):
            count = feature_histograms[fid][bin_i]
            if count > 0:
                hist_bins.append({
                    "bin_center": float(bin_i),
                    "count": int(count),
                })

        features_out[fid] = {
            "feature_id": fid,
            "top_contexts": top_contexts,
            "max_activation": max_act,
            "mean_activation": sum_act / max(1, len(top_contexts)),
            "frequency": frequency,
            "histogram": hist_bins,
        }

    result = {
        "features": features_out,
        "metadata": {
            "n_features": d_sae,
            "total_tokens": total_tokens,
            "top_k": top_k,
            "context_window": context_window,
        },
    }

    return result


def save_top_activations(result: dict, output_path: str) -> None:
    """Save top activations result to JSON."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved top activations to {path}")
