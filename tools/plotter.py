#!/usr/bin/env python3
"""
Plot Tier-A and Tier-B evaluation summaries for thesis figures.

Usage examples:
  # Plot Tier-A and Tier-B macro bars
  python -m tools.plotter \
    --tierA logs/baseline_runs/run_1/eval_tierA.json \
    --tierB logs/baseline_runs/run_1/eval_tierB.json \
    --out logs/baseline_runs/run_1/plots \
    --labelA "Tier-A (τ=0.75)" --labelB "Tier-B"

Outputs (if matplotlib available):
  - macro_tierA.png, macro_tierB.png, macro_combined.png
  - macro_tierA.csv, macro_tierB.csv

The script is read-only; it does not modify evaluation JSONs.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple


A_METRICS = [
    "StepF1",
    "AdjacencyF1",
    "Kendall",
    "ConstraintCoverage",
    "ConstraintAttachmentF1",
    "A_score",
]

B_METRICS = [
    "GraphF1",
    "NEXT_EdgeF1",
    "Logic_EdgeF1",
    "B_score",
]


def _load(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _available(metrics: List[str], macro: Dict[str, Any]) -> List[Tuple[str, float]]:
    out: List[Tuple[str, float]] = []
    for m in metrics:
        v = macro.get(m)
        if v is None:
            continue
        try:
            out.append((m, float(v)))
        except Exception:
            continue
    return out


def _write_csv(rows: List[Tuple[str, float]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for k, v in rows:
            f.write(f"{k},{v:.3f}\n")


def _plot_bar(rows: List[Tuple[str, float]], title: str, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print(f"[plotter] matplotlib not available; skipping plot: {out_path}")
        return

    labels = [k for k, _ in rows]
    values = [v for _, v in rows]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, values, color="steelblue", edgecolor="navy", linewidth=1.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=15, ha="right")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[plotter] wrote {out_path}")


def _plot_combined(rows_a: List[Tuple[str, float]], label_a: str, rows_b: List[Tuple[str, float]], label_b: str, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print(f"[plotter] matplotlib not available; skipping plot: {out_path}")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: A
    ax = axes[0]
    labels_a = [k for k, _ in rows_a]
    values_a = [v for _, v in rows_a]
    bars = ax.bar(labels_a, values_a, color="steelblue", edgecolor="navy", linewidth=1.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(label_a)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis='x', rotation=15)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    # Right: B
    ax = axes[1]
    labels_b = [k for k, _ in rows_b]
    values_b = [v for _, v in rows_b]
    bars = ax.bar(labels_b, values_b, color="darkorange", edgecolor="saddlebrown", linewidth=1.2)
    ax.set_ylim(0.0, 1.05)
    ax.set_title(label_b)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis='x', rotation=15)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[plotter] wrote {out_path}")


def plot_tier_a(eval_path: Path, out_dir: Path, label: str) -> None:
    data = _load(eval_path)
    macro = data.get("macro_avg", {})
    rows = _available(A_METRICS, macro)
    _write_csv(rows, out_dir / "macro_tierA.csv")
    _plot_bar(rows, f"{label}", out_dir / "macro_tierA.png")


def plot_tier_b(eval_path: Path, out_dir: Path, label: str) -> None:
    data = _load(eval_path)
    macro = data.get("macro_avg", {})
    rows = _available(B_METRICS, macro)
    _write_csv(rows, out_dir / "macro_tierB.csv")
    _plot_bar(rows, f"{label}", out_dir / "macro_tierB.png")


def main() -> int:
    ap = argparse.ArgumentParser(description="Plot Tier-A and Tier-B evaluation macro metrics")
    ap.add_argument("--tierA", type=Path, default=None, help="Path to Tier-A eval JSON (eval_tierA.json)")
    ap.add_argument("--tierB", type=Path, default=None, help="Path to Tier-B eval JSON (eval_tierB.json)")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for plots/CSVs")
    ap.add_argument("--labelA", type=str, default="Tier-A", help="Label for Tier-A plot title")
    ap.add_argument("--labelB", type=str, default="Tier-B", help="Label for Tier-B plot title")
    args = ap.parse_args()

    _ensure_out_dir(args.out)

    rows_a: List[Tuple[str, float]] = []
    rows_b: List[Tuple[str, float]] = []

    if args.tierA and args.tierA.exists():
        plot_tier_a(args.tierA, args.out, args.labelA)
        rows_a = _available(A_METRICS, _load(args.tierA).get("macro_avg", {}))
    if args.tierB and args.tierB.exists():
        plot_tier_b(args.tierB, args.out, args.labelB)
        rows_b = _available(B_METRICS, _load(args.tierB).get("macro_avg", {}))

    if rows_a and rows_b:
        _plot_combined(rows_a, args.labelA, rows_b, args.labelB, args.out / "macro_combined.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
