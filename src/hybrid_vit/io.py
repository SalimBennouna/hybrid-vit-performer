from __future__ import annotations

import csv
import json
import os
from typing import Dict, Any, List, Optional


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_summary_csv(path: str, result: Dict[str, Any]) -> None:
    summary = {k: v for k, v in result.items() if k != "history"}
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(summary.keys())
        writer.writerow(summary.values())


def save_history_csv(path: str, history: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "test_loss", "test_acc", "epoch_time"])
        for row in history:
            writer.writerow([
                row["epoch"],
                row["train_loss"],
                row["train_acc"],
                row["test_loss"],
                row["test_acc"],
                row["epoch_time"],
            ])


def save_summary_json(path: str, result: Dict[str, Any]) -> None:
    summary = {k: v for k, v in result.items() if k != "history"}
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def save_config_yaml(path: str, cfg_dict: Dict[str, Any]) -> None:
    import yaml
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False)


def default_style_paths(save_dir: str, is_baseline: bool, arch: str, kernel: str, m: int):
    if is_baseline:
        return (
            os.path.join(save_dir, "summary_baseline.csv"),
            os.path.join(save_dir, "history_baseline.csv")
        )

    return (
        os.path.join(save_dir, f"summary_arch-{arch}_kernel-{kernel}_m-{m}.csv"),
        os.path.join(save_dir, f"history_arch-{arch}_kernel-{kernel}_m-{m}.csv")
    )