#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from hybrid_vit.config import load_config
from hybrid_vit.sweep import run_grid_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a grid sweep from a YAML sweep config.")
    parser.add_argument("--config", type=str, required=True, help="Path to a sweep YAML config.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Where to save CSVs. If omitted, uses <results_dir>/<dataset> from base config.",
    )
    parser.add_argument(
        "--notebook_names",
        action="store_true",
        help="Save files with the same names as the notebook (summary_arch-... etc.).",
    )
    args = parser.parse_args()

    sweep_cfg = load_config(args.config)
    if "base" not in sweep_cfg or "grid" not in sweep_cfg:
        raise ValueError("sweep.py expects a sweep config with 'base' and 'grid' keys.")

    base = sweep_cfg["base"]
    grid = sweep_cfg["grid"]

    save_dir = args.save_dir
    if save_dir is None:
        results_dir = base.get("results_dir", "results")
        dataset = base.get("dataset", "MNIST")
        save_dir = os.path.join(results_dir, dataset)

    _ = run_grid_sweep(
        base_cfg_dict=base,
        grid=grid,
        save_dir=save_dir,
        notebook_style_filenames=args.notebook_names,
    )


if __name__ == "__main__":
    main()