"""
SAE training loop — written entirely from scratch in PyTorch.

Loads precomputed activations from HDF5, trains a SparseAutoencoder
with L1 sparsity penalty and periodic dead-feature resampling.

Usage:
    python -m src.sae.train --config configs/train_medium.yaml
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.hdf5_dataset import HDF5ActivationDataset
from src.sae.loss import sae_loss
from src.sae.model import SparseAutoencoder
from src.sae.resampling import DeadFeatureTracker, resample_dead_features
from src.utils.config import build_argparser, load_config
from src.utils.logging import setup_logger, wandb_init


def train_sae(config: dict) -> None:
    logger = setup_logger("train")

    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Device: {device}")

    # --- Hyperparameters ---
    d_sae = config["d_sae"]
    lambda_l1 = config.get("lambda_l1", 8e-5)
    lr = config.get("lr", 4e-4)
    batch_size = config.get("batch_size", 4096)
    n_steps = config["n_steps"]
    warmup_steps = config.get("warmup_steps", 1000)
    resample_every = config.get("resample_every", 25_000)
    log_every = config.get("log_every", 100)
    save_every = config.get("save_every", 5_000)
    output_dir = Path(config.get("output_dir", "models/sae_default"))
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"d_sae={d_sae}, lambda_l1={lambda_l1:.2e}, lr={lr:.2e}, "
        f"batch_size={batch_size}, n_steps={n_steps}"
    )

    # --- Dataset ---
    activations_path = config["activations_path"]
    mean_path = str(Path(activations_path).with_suffix(".mean.npy"))
    if not Path(mean_path).exists():
        mean_path = None
        logger.info("No mean file found — training without mean-centring")

    dataset = HDF5ActivationDataset(
        h5_path=activations_path,
        mean_path=mean_path,
        cache_in_ram=config.get("cache_in_ram", True),
    )
    d_model = dataset.d_model
    logger.info(f"Dataset: {len(dataset):,} samples, d_model={d_model}")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 2),
        drop_last=True,
        pin_memory=True,
    )
    data_iter = iter(dataloader)

    # --- Model ---
    sae = SparseAutoencoder(d_model=d_model, d_sae=d_sae).to(device)
    n_params = sum(p.numel() for p in sae.parameters())
    logger.info(f"Model has {n_params:,} parameters")

    # --- Optimiser (no weight decay — incompatible with decoder normalisation) ---
    optimizer = Adam(sae.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=0.0)

    # --- Dead feature tracker ---
    dead_tracker = DeadFeatureTracker(d_sae, n_dead_steps=config.get("dead_steps_window", 12_500))

    # --- Wandb ---
    use_wandb = wandb_init(
        project=config.get("wandb_project", "mini-sae-trainer"),
        config=config,
        entity=config.get("wandb_entity"),
    )

    # --- Training loop ---
    logger.info("Starting training...")
    global_step = 0
    start_time = time.time()
    tokens_seen = 0
    best_loss = float("inf")

    pbar = tqdm(total=n_steps, desc="Training")
    while global_step < n_steps:
        # --- LR warmup ---
        if global_step < warmup_steps:
            factor = global_step / max(1, warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = lr * factor

        # --- Get batch ---
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        x = batch.to(device, non_blocking=True)

        # --- Forward ---
        acts, x_recon = sae(x)
        loss_dict = sae_loss(x, x_recon, acts, lambda_l1=lambda_l1)
        total_loss = loss_dict["total_loss"]

        # --- Backward ---
        optimizer.zero_grad()
        total_loss.backward()

        # Clip gradients (optional, helps stability)
        torch.nn.utils.clip_grad_norm_(sae.parameters(), max_norm=1.0)

        optimizer.step()

        # --- Enforce decoder unit-norm constraint ---
        sae.normalize_decoder_()

        # --- Track dead features ---
        dead_tracker.update(acts)

        # --- Periodic resampling ---
        if (global_step + 1) % resample_every == 0 and global_step > 0:
            n_dead = dead_tracker.num_dead()
            logger.info(f"Step {global_step}: {n_dead} dead features — resampling")
            if n_dead > 0:
                resample_dead_features(
                    sae=sae,
                    dead_mask=dead_tracker.get_dead_mask(),
                    dataloader=dataloader,
                    device=device,
                )
                # Reset tracker after resampling
                dead_tracker.reset()
                # Re-normalise after resampling
                sae.normalize_decoder_()

        tokens_seen += x.size(0)

        # --- Logging ---
        if global_step % log_every == 0:
            elapsed = time.time() - start_time
            step_time = elapsed / max(1, global_step + 1)
            l0 = loss_dict["l0_norm"].item()
            ev = loss_dict["explained_variance"].item()
            n_dead = dead_tracker.num_dead()
            current_lr = optimizer.param_groups[0]["lr"]

            log_data = {
                "step": global_step,
                "total_loss": total_loss.item(),
                "l2_loss": loss_dict["l2_loss"].item(),
                "l1_loss": loss_dict["l1_loss"].item(),
                "l0_norm": l0,
                "explained_variance": ev,
                "dead_features": n_dead,
                "lr": current_lr,
                "tokens_seen": tokens_seen,
                "step_time_ms": step_time * 1000,
            }
            pbar.set_postfix(
                loss=f"{total_loss.item():.3f}",
                l0=f"{l0:.1f}",
                ev=f"{ev:.3f}",
                dead=n_dead,
            )

            if use_wandb:
                import wandb

                wandb.log(log_data, step=global_step)

        # --- Checkpoint ---
        if (global_step + 1) % save_every == 0:
            ckpt_path = output_dir / f"sae_step_{global_step + 1}.pt"
            torch.save(
                {
                    "step": global_step + 1,
                    "model_state_dict": sae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config,
                    "loss": total_loss.item(),
                    "l0": loss_dict["l0_norm"].item(),
                    "explained_variance": loss_dict["explained_variance"].item(),
                },
                ckpt_path,
            )
            logger.info(f"Checkpoint saved: {ckpt_path}")

            if total_loss.item() < best_loss:
                best_loss = total_loss.item()
                best_path = output_dir / "sae_best.pt"
                torch.save(
                    {
                        "step": global_step + 1,
                        "model_state_dict": sae.state_dict(),
                        "config": config,
                        "loss": best_loss,
                        "l0": loss_dict["l0_norm"].item(),
                        "explained_variance": loss_dict["explained_variance"].item(),
                    },
                    best_path,
                )

        global_step += 1
        pbar.update(1)

    pbar.close()

    # --- Save final model ---
    final_path = output_dir / "sae_final.pt"
    torch.save(
        {
            "step": global_step,
            "model_state_dict": sae.state_dict(),
            "config": config,
            "loss": total_loss.item(),
            "l0": loss_dict["l0_norm"].item(),
            "explained_variance": loss_dict["explained_variance"].item(),
        },
        final_path,
    )
    logger.info(f"Final model saved: {final_path}")

    elapsed = time.time() - start_time
    logger.info(
        f"Training complete: {global_step} steps in {elapsed:.1f}s "
        f"({elapsed / 60:.1f} min)"
    )

    if use_wandb:
        import wandb
        wandb.finish()


def main() -> None:
    parser = build_argparser()
    parser.add_argument("--activations_path", type=str)
    parser.add_argument("--d_sae", type=int)
    parser.add_argument("--lambda_l1", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--n_steps", type=int)
    parser.add_argument("--resample_every", type=int)
    parser.add_argument("--warmup_steps", type=int)
    parser.add_argument("--wandb_project", type=str)
    parser.add_argument("--wandb_entity", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--cache_in_ram", type=lambda s: s.lower() == "true", default=None)
    parser.add_argument("--num_workers", type=int)

    args = parser.parse_args()

    defaults = dict(
        activations_path="data/activations.h5",
        d_sae=6144,
        lambda_l1=8e-5,
        lr=4e-4,
        batch_size=4096,
        n_steps=200_000,
        resample_every=25_000,
        warmup_steps=1000,
        wandb_project="mini-sae-trainer",
        output_dir="models/sae_medium",
        cache_in_ram=True,
        num_workers=2,
    )
    config = load_config(args.config, args, defaults=defaults)

    train_sae(config)


if __name__ == "__main__":
    main()
