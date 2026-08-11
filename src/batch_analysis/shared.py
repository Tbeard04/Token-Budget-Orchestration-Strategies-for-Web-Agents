from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


# ── Styling ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
COLOURS = {
    "A": "#2196F3", # blue
    "B": "#FF9800", # orange
    "C": "#4CAF50", # green
}

# Terminations caused by running out of budget rather than by agent behaviour
BUDGET_TERMINATIONS = {
    "safety_token_cap", "budget_would_exceed", "budget_exhausted_mid_step",
}
# Terminations caused by the agent getting stuck - these are Stop examples
STUCK_TERMINATIONS = {
    "navigation_cycle", "repeated_action_failure",
}


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


# Loading

def load(path: str) -> pd.DataFrame:
    """Load a JSONL results file, dropping error-only rows."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    df = pd.DataFrame(rows)
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
    return df


def join_tiers(df: pd.DataFrame, tiers_path: str) -> pd.DataFrame:
    """Left-join difficulty tiers and task categories onto episode data."""
    if not tiers_path or not Path(tiers_path).exists():
        return df
    tiers = pd.read_json(tiers_path, lines=True)
    cols = [c for c in ["task_id", "difficulty_tier", "task_category",
                        "rubric_total"] if c in tiers.columns]
    df = df.merge(tiers[cols], on="task_id", how="left")
    matched = df["difficulty_tier"].notna().sum()
    print(f"Joined difficulty tiers: {matched}/{len(df)} episodes matched")
    return df

