import logging
import os
import sys
from typing import Any, Dict, Optional

import wandb


def setup_logger(name: str = "sae") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )
    handler.setFormatter(fmt)

    if not logger.handlers:
        logger.addHandler(handler)

    return logger


def wandb_init(
    project: str = "mini-sae-trainer",
    config: Optional[Dict[str, Any]] = None,
    entity: Optional[str] = None,
) -> bool:
    api_key = os.environ.get("WANDB_API_KEY")
    if not api_key:
        print("[wandb] WANDB_API_KEY not set — skipping wandb logging")
        return False

    try:
        wandb.init(
            project=project,
            entity=entity,
            config=config,
            settings=wandb.Settings(start_method="thread"),
        )
        return True
    except Exception as e:
        print(f"[wandb] Failed to initialise: {e}")
        return False
