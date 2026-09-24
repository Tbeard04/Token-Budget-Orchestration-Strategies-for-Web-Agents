"""
compare_models.py generates figures and one comparison table for the three router versions trained by train_router.py
"""

from __future__ import annotations
 
import argparse
import json
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
 
# validated categorical slots 1-3 (all-pairs safe), one per model
MODEL_COLOUR = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
MODEL_MARKER = {1: "o", 2: "s", 3: "^"}
# ordinal ramp for difficulty tiers (same as tier_diagrams.py)
TIER_COLOUR = {"Easy": "#86b6ef", "Medium": "#2a78d6", "Hard": "#104281"}
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BUDGETS = [2000, 4000, 8000, 16000, 32000, 64000]
 
plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": INK2, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 1, "axes.axisbelow": True, "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE, "legend.frameon": False,
})

#load the metrics and oof predictions for each model
def load(models_dir: Path) -> dict[int, dict]:
    out = {} #output dictionary
    for n in (1, 2, 3):
        d = models_dir / f"model_{n}"
        if not (d / "metrics.json").exists():
            continue
        out[n] = {
            "metrics": json.loads((d / "metrics.json").read_text()),
            "oof": [json.loads(l) for l in (d / "oof_predictions.jsonl").read_text().splitlines() if l.strip()],
        }
    if not out:
        raise SystemExit(f"no model_N/metrics.json under {models_dir}")
    return out
 
 
#create a new figure
def new_fig(title: str, size=(8, 5.2)):
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(top=0.88)
    fig.text(0.02, 0.97, title, fontsize=13, fontweight="bold", color=INK, va="top")
    return fig, ax
 
#save the figure
def save(fig, path: Path) -> None:
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
 
#get the frontier at a given quantile
def at_quantile(m: dict, q: float) -> dict:
    return next(r for r in m["frontier"] if r["quantile"] == q)
 
#get the suggested row
def suggested_row(m: dict) -> dict:
    return next(r for r in m["frontier"] if r["threshold"] == m["suggested_threshold"])

