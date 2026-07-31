from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# Styling
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
})
COLOURS = {
    "A": "#2196F3",   # blue
    "B": "#FF9800",   # orange
    "C": "#4CAF50",   # green (for later)
}

def load(path: str) -> pd.DataFrame:
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
    # Drop error-only rows (crashes, not real episodes)
    if "error" in df.columns:
        df = df[df["error"].isna()].copy()
    return df
