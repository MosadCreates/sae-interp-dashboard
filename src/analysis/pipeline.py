"""Feature analysis orchestrator.

Runs all analysis steps in sequence:
  1. Top-K activating contexts per feature (streaming over corpus)
  2. Feature auto-labelling + search index
  3. Feature geometry analysis

Usage:
    python -m src.analysis.pipeline --config configs/analyze.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.analysis.auto_label import build_feature_index
from src.analysis.geometry import compute_geometry
from src.analysis.top_activations import compute_top_activations, save_top_activations
from src.model import ActivationExtractor
from src.sae.model import SparseAutoencoder
from src.utils.config import build_argparser, load_config
from src.utils.logging import setup_logger
from src.utils.seeds import seed_everything


def run_analysis_pipeline(config: dict) -> None:
    logger = setup_logger("analysis")
    seed_everything(config.get("seed", 42))

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    output_dir = Path(config.get("output_dir", "results"))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = config.get("model_path")
    if model_path is None or not Path(model_path).exists():
        logger.error(f"Model not found at {model_path}")
        return

    # --- Load SAE ---
    logger.info(f"Loading SAE from {model_path}")
    checkpoint = torch.load(model_path, map_location="cpu")
    sae_config = checkpoint.get("config", {})

    d_model = sae_config.get("d_model", 768)
    d_sae = sae_config.get("d_sae", 6144)

    sae = SparseAutoencoder(d_model=d_model, d_sae=d_sae)
    sae.load_state_dict(checkpoint["model_state_dict"])
    sae.to(device)
    sae.eval()
    logger.info(f"SAE loaded: d_model={d_model}, d_sae={d_sae}")

    # --- Load activation extractor ---
    extractor = ActivationExtractor(
        model_name=config.get("model_name", "gpt2"),
        layer=config.get("layer", 8),
        hook_point=config.get("hook_point", "residual_post"),
        device=device,
    )

    # --- Step 1: Top activating contexts ---
    top_acts_path = output_dir / "top_activations.json"
    if top_acts_path.exists() and not config.get("force_recompute", False):
        logger.info(f"Loading existing top activations from {top_acts_path}")
        with open(top_acts_path) as f:
            top_activations = json.load(f)
    else:
        logger.info("Computing top activating contexts (streaming over corpus)...")
        top_activations = compute_top_activations(sae, extractor, config, device)
        save_top_activations(top_activations, str(top_acts_path))

    # --- Step 2: Feature index with auto-labels ---
    index_path = output_dir / "feature_index.json"
    logger.info("Building feature index with auto-labels...")
    build_feature_index(top_activations, str(index_path))

    # --- Step 3: Geometry analysis ---
    logger.info("Computing feature geometry...")
    compute_geometry(sae, str(output_dir))

    # Cleanup
    extractor.remove_hooks()
    logger.info("Analysis pipeline complete.")


def main() -> None:
    parser = build_argparser()
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--layer", type=int)
    parser.add_argument("--hook_point", type=str)
    parser.add_argument("--n_top_tokens", type=int)
    parser.add_argument("--context_window", type=int)
    parser.add_argument("--max_analysis_tokens", type=int)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--force_recompute", action="store_true")

    args = parser.parse_args()

    defaults = dict(
        model_path="models/sae_medium/sae_final.pt",
        model_name="gpt2",
        layer=8,
        hook_point="residual_post",
        n_top_tokens=50,
        context_window=10,
        max_analysis_tokens=5_000_000,
        output_dir="results",
        batch_size=8,
        num_workers=0,
    )
    config = load_config(args.config, args, defaults=defaults)

    run_analysis_pipeline(config)


if __name__ == "__main__":
    main()
