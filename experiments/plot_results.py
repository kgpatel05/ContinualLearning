"""
Plotting for continual-learning logs with:
1) Confusion matrix heatmaps (per dataset/algo/backend) averaged over seeds.
2) Per-class accuracy trajectories (lines start when class first appears).
3) Average accuracy over experiences with backends overlaid.

Usage:
    python experiments/plot_results.py --inputs experiments/results/*.json --outdir experiments/plots
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot confusion matrices and per-class accuracy curves.")
    p.add_argument("--inputs", nargs="+", required=True, help="JSON files or glob patterns (e.g., experiments/results/*.json).")
    p.add_argument("--outdir", type=str, default="experiments/plots", help="Directory to save plots.")
    return p.parse_args()


def load_runs(patterns: Sequence[str]) -> List[Dict]:
    files: List[Path] = []
    for pat in patterns:
        files.extend(Path().glob(pat))
    runs: List[Dict] = []
    seen = set()
    for f in sorted(files):
        if f.suffix != ".json":
            continue
        if f.resolve() in seen:
            continue
        seen.add(f.resolve())
        with open(f, "r", encoding="utf-8") as fh:
            r = json.load(fh)
            runs.append(r)
    return runs


def group_runs(runs: List[Dict]) -> Dict[Tuple[str, str, str], List[Dict]]:
    grouped: Dict[Tuple[str, str, str], List[Dict]] = {}
    for r in runs:
        key = (r.get("dataset", "unknown"), r.get("algo", "unknown"), r.get("backend", "unknown"))
        grouped.setdefault(key, []).append(r)
    return grouped


def pad_sequences(seq: List[List[float]]) -> np.ndarray:
    if not seq:
        return np.empty((0, 0))
    max_len = max(len(s) for s in seq)
    arr = np.full((len(seq), max_len), np.nan, dtype=float)
    for i, s in enumerate(seq):
        arr[i, : len(s)] = s
    return arr


def clean_per_class_list(raw: Sequence[Sequence[float]]) -> List[List[float]]:
    cleaned: List[List[float]] = []
    for row in raw:
        cleaned.append([np.nan if v is None else float(v) for v in row])
    return cleaned


def average_confusions(confusions: List[List[List[List[int]]]]) -> np.ndarray:
    """
    Confusions is a list of runs; each run has a list of conf matrices (one per exp).
    We average the *final* confusion across runs, padding to the max class count.
    """
    mats: List[np.ndarray] = []
    for conf_list in confusions:
        if not conf_list:
            continue
        final = np.array(conf_list[-1], dtype=float)
        if final.ndim != 2 or final.size == 0:
            continue
        mats.append(final)

    if not mats:
        return np.empty((0, 0))

    max_rows = max(m.shape[0] for m in mats)
    max_cols = max(m.shape[1] for m in mats)
    agg = np.zeros((max_rows, max_cols), dtype=float)

    for m in mats:
        tmp = np.zeros_like(agg)
        tmp[: m.shape[0], : m.shape[1]] = m
        agg += tmp

    # normalize rows to probabilities (avoid divide-by-zero)
    row_sums = agg.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return agg / row_sums


def plot_confusions(groups: Dict[Tuple[str, str, str], List[Dict]], outdir: Path) -> None:
    for (dataset, algo, backend), runs in groups.items():
        confs = [r.get("confusions", []) for r in runs]
        conf = average_confusions(confs)
        if conf.size == 0:
            continue
        plt.figure(figsize=(6, 5))
        im = plt.imshow(conf, cmap="viridis", origin="lower", aspect="auto")
        plt.colorbar(im, label="Normalized count")
        plt.xlabel("Predicted class")
        plt.ylabel("True class")
        plt.title(f"{dataset} | {algo} | {backend} (final confusion avg over seeds)")
        out = outdir / f"{dataset}_{algo}_{backend}_confusion.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()


def plot_per_class(groups: Dict[Tuple[str, str, str], List[Dict]], outdir: Path) -> None:
    for (dataset, algo, backend), runs in groups.items():
        # Collect per-class per-exp accuracies across seeds
        per_class_lists: List[List[List[float]]] = [clean_per_class_list(r.get("per_class_acc_by_exp", [])) for r in runs]
        non_empty = [seq for seq in per_class_lists if seq]
        if not non_empty:
            continue
        max_exps = max(len(seq) for seq in non_empty)
        max_classes = max(len(seq[0]) for seq in non_empty if seq)

        # Mean over seeds for each class/exp
        mean_acc = np.full((max_exps, max_classes), np.nan, dtype=float)
        for exp_idx in range(max_exps):
            for cls in range(max_classes):
                vals = []
                for seq in non_empty:
                    if exp_idx < len(seq) and cls < len(seq[exp_idx]):
                        val = seq[exp_idx][cls]
                        if val == val:  # not nan
                            vals.append(val)
                if vals:
                    mean_acc[exp_idx, cls] = float(np.mean(vals))

        plt.figure(figsize=(7, 5))
        exps = np.arange(max_exps)
        for cls in range(max_classes):
            y = mean_acc[:, cls]
            if np.all(np.isnan(y)):
                continue
            first_valid = np.argmax(~np.isnan(y))
            plt.plot(exps[first_valid:], y[first_valid:], label=f"class {cls}", linewidth=1.6)
        plt.xlabel("Experience")
        plt.ylabel("Per-class accuracy (%)")
        plt.title(f"{dataset} | {algo} | {backend}")
        plt.grid(True, alpha=0.3)
        plt.legend(loc="best", ncol=2, fontsize=8)
        out = outdir / f"{dataset}_{algo}_{backend}_per_class.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()


def plot_avg_accuracy_overlay(runs: List[Dict], outdir: Path) -> None:
    if not runs:
        return
    by_pair: Dict[Tuple[str, str], Dict[str, np.ndarray]] = {}
    for r in runs:
        pair = (r.get("dataset", "unknown"), r.get("algo", "unknown"))
        backend = r.get("backend", "unknown")
        curve = np.array(r.get("per_exp_acc", []), dtype=float)
        by_pair.setdefault(pair, {}).setdefault(backend, []).append(curve)

    colors = {"clx": "#1f77b4", "avalanche": "#ff7f0e"}
    for pair, back_map in by_pair.items():
        plt.figure(figsize=(6, 4))
        for backend, curves in back_map.items():
            arr = pad_sequences([c.tolist() for c in curves])
            if arr.size == 0:
                continue
            mean = np.nanmean(arr, axis=0)
            std = np.nanstd(arr, axis=0)
            xs = np.arange(len(mean))
            color = colors.get(backend, None)
            plt.plot(xs, mean, label=f"{backend} mean", color=color, linewidth=2.2)
            plt.fill_between(xs, mean - std, mean + std, color=color, alpha=0.2)
        plt.xlabel("Experience")
        plt.ylabel("Accuracy (%)")
        plt.title(f"{pair[0]} | {pair[1]} | Avg accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        out = outdir / f"{pair[0]}_{pair[1]}_avg_accuracy_overlay.png"
        plt.tight_layout()
        plt.savefig(out, dpi=200)
        plt.close()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.inputs)
    if not runs:
        print("No runs loaded; check --inputs patterns.")
        return

    groups = group_runs(runs)
    plot_confusions(groups, outdir)
    plot_per_class(groups, outdir)
    plot_avg_accuracy_overlay(runs, outdir)
    print(f"Plots written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
