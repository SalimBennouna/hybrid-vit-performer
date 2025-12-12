from __future__ import annotations

from typing import Dict, Any, Optional

import torch
import torch.nn as nn

from .config import TrainConfig
from .data import get_dataloaders
from .model.hybrid_vit import HybridViT
from .train import fit
from .benchmark import measure_inference_time
from .io import save_summary_csv, save_history_csv, default_style_paths, ensure_dir


def run_single_experiment(
    cfg: TrainConfig,
    save_dir: Optional[str] = None,
    notebook_style_filenames: bool = True,
    is_baseline: Optional[bool] = None,
) -> Dict[str, Any]:

    device = torch.device(cfg.device)

    train_loader, test_loader, in_channels = get_dataloaders(cfg)
    model = HybridViT(cfg, in_channels=in_channels).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    print(f"\n=== Dataset={cfg.dataset} | Arch={cfg.architecture} | "
          f"Kernel={cfg.kernel_type} | m={cfg.m_features} ===")

    history, best_test_acc, total_train_time = fit(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        epochs=cfg.epochs
    )

    inf_mean, inf_std = measure_inference_time(model, test_loader, device)
    print(f"Best Test Acc: {best_test_acc:.2f}%")
    print(f"Inference: {inf_mean*1000:.2f} ± {inf_std*1000:.2f} ms / batch")

    result: Dict[str, Any] = {
        "dataset": cfg.dataset,
        "architecture": cfg.architecture,
        "kernel_type": cfg.kernel_type,
        "m_features": cfg.m_features,
        "best_test_acc": best_test_acc,
        "total_train_time": total_train_time,
        "avg_train_time_per_epoch": total_train_time / cfg.epochs,
        "infer_time_per_batch_mean_ms": inf_mean * 1000,
        "infer_time_per_batch_std_ms": inf_std * 1000,
        "history": history,
    }

    if save_dir is not None:
        ensure_dir(save_dir)
        if is_baseline is None:
            is_baseline = (cfg.architecture == "all_standard")

        if notebook_style_filenames:
            summary_path, history_path = default_style_paths(
                save_dir, is_baseline=is_baseline, arch=cfg.architecture, kernel=cfg.kernel_type, m=cfg.m_features
            )
        else:
            # generic
            summary_path = f"{save_dir}/summary.csv"
            history_path = f"{save_dir}/history.csv"

        save_summary_csv(summary_path, result)
        save_history_csv(history_path, history)

        print("\nSummary saved to:", summary_path)
        print("History CSV saved to:", history_path)

    return result