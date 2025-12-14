from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, Literal, Optional

import torch
import yaml


def resolve_device(device: str) -> str:
    if device != "auto":
        return device

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class TrainConfig:
    dataset: Literal["MNIST", "CIFAR10"] = "MNIST"
    img_size: int = 28          # 32 for CIFAR-10
    patch_size: int = 4
    num_classes: int = 10
    dim: int = 64               # projection dimension
    depth: int = 4              # number of transformer layers
    num_heads: int = 4
    mlp_ratio: float = 2.0      # hidden dimension = dim * mlp_ratio
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 50
    m_features: int = 256       # Performer random feature dimension m
    architecture: Literal[
        "all_standard",
        "all_performer",
        "intertwined",
        "performer_first",
        "standard_first",
    ] = "intertwined"
    kernel_type: Literal["relu", "softmax"] = "relu"
    device: str = "auto"      

    results_dir: str = "results"
    run_name: Optional[str] = None  
    seed: Optional[int] = None      


def _deep_update(base: Dict[str, Any], upd: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str) -> Dict[str, Any]:
    """
    Loads YAML into a plain dict. Used by scripts.
    Supports either:
      - baseline config (flat keys) OR
      - sweep config with {"base": {...}, "grid": {...}}
    """
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


def config_from_dict(d: Dict[str, Any]) -> TrainConfig:
    """
    Create TrainConfig from a dict; keeps defaults for missing keys.
    """
    cfg = TrainConfig(**{k: v for k, v in d.items() if hasattr(TrainConfig, k)})
    # Defensive casts in case numeric fields arrive as strings (e.g., from CLI or YAML parsing).
    for name in ["lr", "weight_decay", "mlp_ratio"]:
        val = getattr(cfg, name, None)
        if isinstance(val, str):
            try:
                setattr(cfg, name, float(val))
            except ValueError:
                pass
    for name in ["epochs", "batch_size", "m_features", "img_size", "patch_size", "dim", "depth", "num_heads", "num_classes"]:
        val = getattr(cfg, name, None)
        if isinstance(val, str):
            try:
                setattr(cfg, name, int(float(val)))
            except ValueError:
                pass
    if isinstance(cfg.seed, str):
        try:
            cfg.seed = int(cfg.seed)
        except ValueError:
            pass
    cfg.device = resolve_device(cfg.device)
    return cfg
