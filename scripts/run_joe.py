#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure repository root and src/ are on sys.path so hybrid_vit can be imported when run as a script.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (SRC, ROOT):
    p_str = str(p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from hybrid_vit.config import config_from_dict, load_config  # noqa: E402
from hybrid_vit.experiment import run_single_experiment  # noqa: E402


KERNS = ["relu", "softmax"]
M_LIST = [16, 32, 64, 128, 256]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MNIST intertwined experiments (subset) for specific kernels/m values.")
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Device to use (overrides config). Default: mps.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Override output directory (default: results/MNIST).",
    )
    args = parser.parse_args()

    base_cfg = load_config("configs/mnist.yaml")
    base_cfg["dataset"] = "MNIST"
    if args.device is not None:
        base_cfg["device"] = args.device

    results_root = base_cfg.get("results_dir", "results")
    save_dir = args.save_dir or os.path.join(results_root, "MNIST")

    for kernel in KERNS:
        for m in M_LIST:
            merged = dict(base_cfg)
            merged.update(
                {
                    "architecture": "intertwined",
                    "kernel_type": kernel,
                    "m_features": m,
                    "seed": args.seed,
                }
            )
            cfg = config_from_dict(merged)
            run_single_experiment(
                cfg,
                save_dir=save_dir,
                notebook_style_filenames=True,
                is_baseline=False,
            )


if __name__ == "__main__":
    main()
