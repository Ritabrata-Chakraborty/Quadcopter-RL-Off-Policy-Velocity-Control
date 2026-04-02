#!/usr/bin/env python3
"""Compare training metrics across multiple experiments.

Recursively searches experiments/ for all tensorboard logs, extracts metrics,
and creates comparison plots where each subplot shows the same metric from
all runs with consistent colors per experiment.

Usage:
    python3 compare.py
    python3 compare.py --pattern "DDPG*"
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use('Agg')

from utils import compute_lowess, extract_tensorboard_metrics


# ------------------------------------------------------------------
# Experiment discovery and organization
# ------------------------------------------------------------------

def find_tensorboard_dirs(experiments_root: str, pattern: str = "*") -> dict[str, str]:
    """Recursively find all tensorboard directories under ``experiments_root``.

    Returns a dict mapping experiment name to tensorboard directory path.
    """
    root = Path(experiments_root)
    tensorboard_dirs: dict[str, str] = {}
    if not root.exists():
        print(f"Experiments directory not found: {root}")
        return tensorboard_dirs
    for tb_dir in root.glob(f'{pattern}/**/tensorboard'):
        if tb_dir.exists() and any(tb_dir.iterdir()):
            exp_name = tb_dir.parts[-3]
            tensorboard_dirs[exp_name] = str(tb_dir)
    return tensorboard_dirs


def extract_all_metrics(
    tensorboard_dirs: dict[str, str],
) -> tuple[dict[str, dict], list[str], dict[str, str]]:
    """Extract metrics from all tensorboard directories.

    Returns:
        all_metrics: ``{metric_name: {exp_name: (steps, values)}}``
        run_names: sorted list of experiment names
        color_map: experiment name -> hex color string
    """
    print(f"Found {len(tensorboard_dirs)} experiment(s): {', '.join(sorted(tensorboard_dirs.keys()))}")

    all_metrics: dict[str, dict] = {}
    run_names = sorted(tensorboard_dirs.keys())

    colors = [
        '#1565C0', '#E53935', '#00838F', '#F57C00', '#6A1B9A',
        '#00897B', '#EF6C00', '#1976D2', '#C62828', '#004D40',
    ]
    color_map = {name: colors[i % len(colors)] for i, name in enumerate(run_names)}

    for exp_name, tb_dir in sorted(tensorboard_dirs.items()):
        print(f"  Extracting {exp_name} ...")
        metrics = extract_tensorboard_metrics(tb_dir)
        for metric_name, values in metrics.items():
            if not values:
                continue
            if metric_name not in all_metrics:
                all_metrics[metric_name] = {}
            steps, vals = zip(*values)
            all_metrics[metric_name][exp_name] = (np.array(steps), np.array(vals))

    return all_metrics, run_names, color_map


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def save_comparison_plot(
    metrics: dict[str, dict],
    run_names: list[str],
    color_map: dict[str, str],
    output_dir: str,
    plot_name: str,
    ylabel: str,
) -> None:
    """Create and save a comparison plot; one subplot per metric, one line per run."""
    n_metrics = len(metrics)
    if n_metrics == 0:
        return

    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
    fig.suptitle(f'Training Comparison — {plot_name}', fontsize=16, fontweight='bold')

    if n_metrics == 1:
        axes = np.array([axes])
    elif rows == 1:
        axes = axes.reshape(1, -1)

    metric_idx = 0
    for metric_name in sorted(metrics.keys()):
        runs_data = metrics[metric_name]
        row, col = metric_idx // cols, metric_idx % cols
        ax = axes[row, col] if axes.ndim > 1 else axes[metric_idx]

        for run_name in run_names:
            if run_name not in runs_data:
                continue
            steps, vals = runs_data[run_name]
            color = color_map[run_name]
            ax.plot(steps, vals, linewidth=0.5, alpha=0.15, color=color)
            ax.plot(steps, compute_lowess(steps, vals, frac=0.1),
                    linewidth=2.5, label=run_name, color=color)

        ax.set_xlabel('Episode')
        ax.set_ylabel(ylabel)
        ax.set_title(metric_name, fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')
        metric_idx += 1

    for i in range(metric_idx, rows * cols):
        row, col = i // cols, i % cols
        ax = axes[row, col] if axes.ndim > 1 else axes[i]
        ax.set_visible(False)

    fig.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'{plot_name}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved: {plot_path}")


def create_comparison_plots(
    all_metrics: dict[str, dict],
    run_names: list[str],
    color_map: dict[str, str],
    output_dir: str,
) -> None:
    """Create loss and performance comparison plots for all metrics."""
    losses_metrics: dict[str, dict] = {}
    perf_metrics: dict[str, dict] = {}
    for metric_name, runs_data in all_metrics.items():
        if not runs_data:
            continue
        if 'Losses' in metric_name:
            losses_metrics[metric_name.replace('Losses/', '')] = runs_data
        elif 'Perf' in metric_name:
            perf_metrics[metric_name.replace('Perf/', '')] = runs_data

    if losses_metrics:
        save_comparison_plot(losses_metrics, run_names, color_map, output_dir,
                             'Training_Losses_Comparison', 'Loss')
    if perf_metrics:
        save_comparison_plot(perf_metrics, run_names, color_map, output_dir,
                             'Training_Performance_Comparison', 'Value')


# ------------------------------------------------------------------
# Summary report
# ------------------------------------------------------------------

def create_summary_report(
    all_metrics: dict[str, dict],
    run_names: list[str],
    output_dir: str,
) -> None:
    """Write a JSON summary of final metric values across all runs."""
    losses: dict[str, int] = {}
    perf: dict[str, int] = {}
    for metric_name, runs_data in sorted(all_metrics.items()):
        if 'Losses' in metric_name:
            losses[metric_name.replace('Losses/', '')] = len(runs_data)
        elif 'Perf' in metric_name:
            perf[metric_name.replace('Perf/', '')] = len(runs_data)

    run_details: dict[str, dict] = {}
    for run_name in run_names:
        run_details[run_name] = {"metrics": {}}
        for metric_name in sorted(all_metrics.keys()):
            if run_name not in all_metrics[metric_name]:
                continue
            steps, vals = all_metrics[metric_name][run_name]
            clean = metric_name.replace('Losses/', '').replace('Perf/', '')
            run_details[run_name]["metrics"][clean] = {
                "num_points": int(len(steps)),
                "final_value": float(vals[-1]),
            }

    summary = {
        "experiments_compared": run_names,
        "metrics_available": {"losses": losses, "performance": perf},
        "run_details": run_details,
    }
    report_path = os.path.join(output_dir, 'summary.json')
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved: {report_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description='Compare training metrics across multiple experiments.'
    )
    parser.add_argument(
        '--pattern', default='*',
        help='Glob pattern to match experiment names (default: all)',
    )
    args = parser.parse_args()

    experiments_dir = 'experiments'
    output_dir = 'experiments/comparison'
    tensorboard_dirs = find_tensorboard_dirs(experiments_dir, args.pattern)

    if not tensorboard_dirs:
        print("No tensorboard logs found.")
        return

    all_metrics, run_names, color_map = extract_all_metrics(tensorboard_dirs)
    if not all_metrics:
        print("No metrics found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    create_comparison_plots(all_metrics, run_names, color_map, output_dir)
    create_summary_report(all_metrics, run_names, output_dir)
    print(f"Done: {output_dir}/")


if __name__ == "__main__":
    main()
