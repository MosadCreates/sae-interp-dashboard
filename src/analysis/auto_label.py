"""Auto-label SAE features and build keyword search index.

Strategies:
  1. Most frequent token among top-K activating examples.
  2. Heuristic: capitalisation, punctuation, common patterns.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


def auto_label_feature(
    top_contexts: List[dict],
    top_tokens: Optional[List[str]] = None,
) -> str:
    """Generate a human-readable label for a feature.

    Strategy:
      1. If top_tokens provided, find the most semantically meaningful one.
      2. Otherwise, find the most frequent non-trivial token in the contexts.
      3. Fall back to heuristics (punctuation, capitalisation, etc.)
    """
    # Collect all tokens from contexts
    all_tokens: List[str] = []
    for ctx in top_contexts:
        all_tokens.extend(ctx.get("tokens", []))

    if top_tokens:
        all_tokens.extend(top_tokens)

    if not all_tokens:
        return "no_data"

    # Filter out trivial tokens
    meaningful = [
        t
        for t in all_tokens
        if t and len(t) > 1 and t not in ("", " ", "\n", ".", ",", "!", "?", ";", ":", "\t")
    ]

    if not meaningful:
        # Use single-character tokens
        meaningful = [t for t in all_tokens if t and t.strip()]

    if not meaningful:
        return "empty"

    # Most common
    counter = Counter(meaningful)
    most_common = counter.most_common(1)[0][0]

    # Clean up
    label = most_common.replace("Ġ", "").replace("Ċ", "\\n").strip()
    if not label:
        label = "token_" + most_common

    return label[:40]  # cap length


def build_feature_index(
    top_activations: dict,
    output_path: str,
) -> dict:
    """Build a searchable feature index with auto-labels.

    Args:
        top_activations: the dict loaded from top_activations.json.
        output_path: where to save the index JSON.

    Returns:
        The index dict (also saved to disk).
    """
    features = top_activations.get("features", {})

    index: Dict[str, Any] = {
        "features": {},
        "search_terms": {},
        "metadata": {
            "n_features": len(features),
        },
    }

    for fid_str, fdata in features.items():
        fid = int(fid_str)
        # Extract top tokens from contexts
        top_tokens = []
        for ctx in fdata.get("top_contexts", []):
            for t in ctx.get("tokens", []):
                cleaned = t.replace("Ġ", "").replace("Ċ", "\\n").strip()
                if cleaned and len(cleaned) <= 50:
                    top_tokens.append(cleaned.lower())

        # Deduplicate while preserving order
        seen: set = set()
        unique_tokens = []
        for t in top_tokens:
            if t not in seen:
                seen.add(t)
                unique_tokens.append(t)

        # Generate label
        label = auto_label_feature(
            fdata.get("top_contexts", []),
            top_tokens=unique_tokens[:10],
        )

        index["features"][fid_str] = {
            "feature_id": fid,
            "label": label,
            "top_tokens": unique_tokens[:20],
            "frequency": fdata.get("frequency", 0.0),
            "max_activation": fdata.get("max_activation", 0.0),
        }

        # Build search index
        for token in unique_tokens:
            if token not in index["search_terms"]:
                index["search_terms"][token] = []
            index["search_terms"][token].append(fid)

    # Save
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(index, f, indent=2)
    print(f"Saved feature index to {path}")

    return index


def keyword_search(
    keyword: str,
    feature_index: dict,
    max_results: int = 20,
) -> List[dict]:
    """Search features by keyword.

    Matches against auto-labels and top activating tokens.
    Returns list of feature summary dicts.
    """
    keyword_lower = keyword.lower().strip()
    if not keyword_lower:
        return []

    search_terms = feature_index.get("search_terms", {})
    features = feature_index.get("features", {})

    matched_fids: set = set()

    # Direct match
    for term, fids in search_terms.items():
        if keyword_lower in term:
            matched_fids.update(fids)

    # Partial match in labels
    for fid_str, fdata in features.items():
        label = fdata.get("label", "").lower()
        if keyword_lower in label or keyword_lower in label.replace("\\n", "\n"):
            matched_fids.add(int(fid_str))

    # Score and rank
    scored = []
    for fid in matched_fids:
        fdata = features.get(str(fid), {})
        score = 0
        label = fdata.get("label", "").lower()
        if keyword_lower == label:
            score += 100
        if keyword_lower in label:
            score += 50
        for t in fdata.get("top_tokens", []):
            if keyword_lower == t:
                score += 30
            elif keyword_lower in t:
                score += 10

        scored.append({
            "feature_id": fid,
            "label": fdata.get("label", ""),
            "top_tokens": fdata.get("top_tokens", [])[:5],
            "frequency": fdata.get("frequency", 0.0),
            "score": score,
        })

    scored.sort(key=lambda x: -x["score"])
    return scored[:max_results]
