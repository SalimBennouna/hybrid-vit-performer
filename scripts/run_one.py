#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from pathlib import Path

# Ensure repository root and src/ are on sys.path so hybrid_vit can be imported when run as a script.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (SRC, ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from hybrid_vit.config import TrainConfig, config_from_dict, load_config  # noqa: E402
from hybrid_vit.experiment import run_single_experiment  # noqa: E402

ARCH_CHOICES = [
    "all_standard",
    "all_performer",
    "intertwined",
    "performer_first",
    "standard_first",
]
KERNEL_CHOICES = ["relu", "softmax"]

DEFAULT_BASE = {
    "MNIST": "configs/mnist.yaml",
    "CIFAR10": "configs/cifar10.yaml",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single HybridViT experiment with CLI overrides.")
    parser.add_argument(
        "--dataset",
        choices=["MNIST", "CIFAR10"],
        required=True,
        help="Dataset to run.",
    )
    parser.add_argument(
        "--architecture",
        choices=ARCH_CHOICES,
        required=True,
        help="Model architecture to use.",
    )
    parser.add_argument(
        "--kernel_type",
        choices=KERNEL_CHOICES,
        required=True,
        help="Performer kernel.",
    )
    parser.add_argument(
        "--m_features",
        type=int,
        required=True,
        help="Number of random features (m).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for full reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use (overrides config).",
    )
    args = parser.parse_args()

    base_cfg_path = DEFAULT_BASE[args.dataset]
    base_cfg = load_config(base_cfg_path)

    # Override with CLI choices
    base_cfg.update(
        {
            "dataset": args.dataset,
            "architecture": args.architecture,
            "kernel_type": args.kernel_type,
            "m_features": args.m_features,
            "seed": args.seed,
            "device": args.device or base_cfg.get("device", "auto"),
        }
    )

    train_cfg: TrainConfig = config_from_dict(base_cfg)

    results_root = base_cfg.get("results_dir", "results")
    save_dir = os.path.join(results_root, args.dataset)

    # Log fixed hyperparameters (exclude dataset/architecture/kernel/m).
    cfg_view = {k: v for k, v in asdict(train_cfg).items() if k not in {"dataset", "architecture", "kernel_type", "m_features"}}
    print("\n--- Training configuration (fixed params) ---")
    for k in sorted(cfg_view):
        print(f"{k}: {cfg_view[k]}")
    print("--------------------------------------------\n")

    is_baseline = train_cfg.architecture == "all_standard"
    _ = run_single_experiment(
        train_cfg,
        save_dir=save_dir,
        notebook_style_filenames=True,
        is_baseline=is_baseline,
    )


if __name__ == "__main__":
    main()
