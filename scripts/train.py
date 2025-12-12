#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from hybrid_vit.config import load_config, config_from_dict, TrainConfig
from hybrid_vit.experiment import run_single_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one HybridViT experiment from a YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to a baseline YAML config.")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=None,
        help="Where to save CSVs. If omitted, uses <results_dir>/<dataset> from config.",
    )
    parser.add_argument(
        "--notebook_names",
        action="store_true",
        help="Save files with the same names as the notebook (summary_baseline.csv, etc.).",
    )
    args = parser.parse_args()

    cfg_dict = load_config(args.config)
    if "base" in cfg_dict or "grid" in cfg_dict:
        raise ValueError("train.py expects a baseline config (flat YAML), not a sweep config.")

    cfg: TrainConfig = config_from_dict(cfg_dict)

    save_dir = args.save_dir
    if save_dir is None:
        save_dir = os.path.join(cfg.results_dir, cfg.dataset)

    is_baseline = (cfg.architecture == "all_standard")

    print(f"Device:{cfg.device}")
    _ = run_single_experiment(
        cfg,
        save_dir=save_dir,
        notebook_style_filenames=args.notebook_names,
        is_baseline=is_baseline,
    )


if __name__ == "__main__":
    main()