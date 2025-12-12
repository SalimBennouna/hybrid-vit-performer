from __future__ import annotations

import os
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt


def load_history(save_dir: str, arch: str, kernel: str, m: int) -> Optional[pd.DataFrame]:
    fname = f"history_arch-{arch}_kernel-{kernel}_m-{m}.csv"
    path = os.path.join(save_dir, fname)
    if not os.path.exists(path):
        print("Missing:", path)
        return None
    return pd.read_csv(path)


def load_baseline(save_dir: str) -> Optional[pd.DataFrame]:
    path = os.path.join(save_dir, "history_baseline.csv")
    if not os.path.exists(path):
        print("Missing baseline:", path)
        return None
    return pd.read_csv(path)


def plot_metric(
    save_dir: str,
    metric: str,
    title: str,
    architectures: List[str],
    kernel_types: List[str],
    m_list: List[int],
    include_baseline: bool = True
) -> None:
    plt.figure(figsize=(10, 6))

    baseline_df = load_baseline(save_dir) if include_baseline else None
    if baseline_df is not None:
        plt.plot(
            baseline_df["epoch"],
            baseline_df[metric],
            label="baseline",
            linewidth=2.5,
            color="black"
        )

    for arch in architectures:
        for kernel in kernel_types:
            for m in m_list:
                df = load_history(save_dir, arch, kernel, m)
                if df is None:
                    continue
                label = f"{arch} | {kernel} | m={m}"
                plt.plot(df["epoch"], df[metric], label=label)

    plt.title(title, fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend(fontsize=8)
    plt.grid(True)
    plt.show()


def plot_m_sweep(
    save_dir: str,
    metric: str,
    arch: str,
    kernel: str,
    m_list: List[int]
) -> None:
    plt.figure(figsize=(8, 5))

    for m in m_list:
        df = load_history(save_dir, arch, kernel, m)
        if df is None:
            continue
        plt.plot(df["epoch"], df[metric], label=f"m={m}")

    plt.title(f"{arch} | {kernel} — {metric}", fontsize=14)
    plt.xlabel("Epoch")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    plt.show()