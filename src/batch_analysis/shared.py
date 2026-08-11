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