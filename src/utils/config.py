import argparse
import os
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(
    config_path: Optional[str] = None,
    cli_args: Optional[argparse.Namespace] = None,
    defaults: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}

    if config_path is not None:
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(path) as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                cfg = loaded

    if defaults is not None:
        for k, v in defaults.items():
            cfg.setdefault(k, v)

    if cli_args is not None:
        overrides = {k: v for k, v in vars(cli_args).items() if v is not None}
        _deep_merge(cfg, overrides)

    cfg.setdefault("seed", 42)
    cfg.setdefault("device", "cuda" if torch_available() else "cpu")

    return cfg


def _deep_merge(base: Dict, overrides: Dict) -> None:
    for k, v in overrides.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def torch_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")
    return parser
