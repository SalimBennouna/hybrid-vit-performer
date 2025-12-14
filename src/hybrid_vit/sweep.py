from __future__ import annotations

from itertools import product
from typing import Dict, Any, List, Optional

from .config import TrainConfig, config_from_dict
from .experiment import run_single_experiment


def run_grid_sweep(
    base_cfg_dict: Dict[str, Any],
    grid: Dict[str, List[Any]],
    save_dir: str,
    notebook_style_filenames: bool = True,
) -> List[Dict[str, Any]]:

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    all_results: List[Dict[str, Any]] = []

    for combo in product(*values):
        combo_dict = dict(zip(keys, combo))
        merged = dict(base_cfg_dict)
        merged.update(combo_dict)

        cfg: TrainConfig = config_from_dict(merged)

        is_baseline = (cfg.architecture == "all_standard")

        result = run_single_experiment(
            cfg,
            save_dir=save_dir,
            notebook_style_filenames=notebook_style_filenames,
            is_baseline=is_baseline
        )
        all_results.append(result)

    return all_results
