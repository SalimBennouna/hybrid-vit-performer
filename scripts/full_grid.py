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

from hybrid_vit.config import load_config, config_from_dict  # noqa: E402
from hybrid_vit.experiment import run_single_experiment  # noqa: E402
from hybrid_vit.sweep import run_grid_sweep  # noqa: E402

ARCHITECTURES = [
    "all_standard",     # baseline ViT
    "intertwined",
    "performer_first",
    "standard_first",
    "all_performer",
]

M_LIST = [8, 16, 32, 64, 128, 256]
KERNEL_TYPES = ["relu", "softmax"]

DEFAULT_BASE = {
    "MNIST": "configs/mnist.yaml",
    "CIFAR10": "configs/cifar10.yaml",
}

def main() -> None:
    parser = argparse.ArgumentParser(description="Run full HybridViT grid.")
    parser.add_argument(
        "--dataset",
        choices=["MNIST", "CIFAR10"],
        required=True,
        help="Dataset to run.",
    )
    parser.add_argument(
        "--base_config",
        type=str,
        default=None,
        help="Path to a baseline YAML config (defaults to the dataset baseline).",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override output directory (default: results/<dataset>).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use (overrides config).",
    )
    args = parser.parse_args()

    base_cfg_path = args.base_config or DEFAULT_BASE[args.dataset]
    base_cfg = load_config(base_cfg_path)

    # Ensure dataset matches the CLI choice
    base_cfg["dataset"] = args.dataset
    if args.device is not None:
        base_cfg["device"] = args.device
    results_root = base_cfg.get("results_dir", "results")
    save_dir = args.save_dir or os.path.join(results_root, args.dataset)

    # Run the baseline once (all_standard) without sweeping m or kernel_type.
    baseline_cfg = dict(base_cfg)
    baseline_cfg["architecture"] = "all_standard"
    baseline_train_cfg = config_from_dict(baseline_cfg)

    cfg_view = {k: v for k, v in asdict(baseline_train_cfg).items() if k not in {"dataset", "architecture", "kernel_type", "m_features"}}
    print("\n--- Baseline configuration (fixed params) ---")
    for k in sorted(cfg_view):
        print(f"{k}: {cfg_view[k]}")
    print("---------------------------------------------\n")

    run_single_experiment(
        baseline_train_cfg,
        save_dir=save_dir,
        notebook_style_filenames=True,
        is_baseline=True,
    )

    grid = {
        "architecture": [arch for arch in ARCHITECTURES if arch != "all_standard"],
        "kernel_type": KERNEL_TYPES,
        "m_features": M_LIST,
    }

    run_grid_sweep(
        base_cfg_dict=base_cfg,
        grid=grid,
        save_dir=save_dir,
        notebook_style_filenames=True,
    )

if __name__ == "__main__":
    main()
