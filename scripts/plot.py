#!/usr/bin/env python3
from __future__ import annotations

import argparse

from hybrid_vit.plotting import plot_metric, plot_m_sweep


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved histories from CSV files.")
    parser.add_argument("--save_dir", type=str, required=True, help="Directory containing history_*.csv files.")

    subparsers = parser.add_subparsers(dest="cmd", required=True)

    # plot_metric command
    p_metric = subparsers.add_parser("metric", help="Plot a metric across many configs (+ baseline).")
    p_metric.add_argument("--metric", type=str, required=True, choices=["train_loss", "train_acc", "test_loss", "test_acc", "epoch_time"])
    p_metric.add_argument("--title", type=str, required=True)

    p_metric.add_argument("--architectures", nargs="+", required=True)
    p_metric.add_argument("--kernel_types", nargs="+", required=True, choices=["relu", "softmax"])
    p_metric.add_argument("--m_list", nargs="+", required=True, type=int)
    p_metric.add_argument("--no_baseline", action="store_true")

    # plot_m_sweep command
    p_msweep = subparsers.add_parser("m_sweep", help="Fix arch+kernel, compare different m values.")
    p_msweep.add_argument("--metric", type=str, required=True, choices=["train_loss", "train_acc", "test_loss", "test_acc", "epoch_time"])
    p_msweep.add_argument("--arch", type=str, required=True)
    p_msweep.add_argument("--kernel", type=str, required=True, choices=["relu", "softmax"])
    p_msweep.add_argument("--m_list", nargs="+", required=True, type=int)

    args = parser.parse_args()

    if args.cmd == "metric":
        plot_metric(
            save_dir=args.save_dir,
            metric=args.metric,
            title=args.title,
            architectures=args.architectures,
            kernel_types=args.kernel_types,
            m_list=args.m_list,
            include_baseline=not args.no_baseline,
        )

    elif args.cmd == "m_sweep":
        plot_m_sweep(
            save_dir=args.save_dir,
            metric=args.metric,
            arch=args.arch,
            kernel=args.kernel,
            m_list=args.m_list,
        )


if __name__ == "__main__":
    main()