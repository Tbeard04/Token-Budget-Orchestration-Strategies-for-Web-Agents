"""
tier_diagrams.py - figures for the difficulty annotation
"""

from __future__ import annotations
 
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
 
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, PathPatch
from matplotlib.path import Path as MPath
 
from rubric import DIMENSIONS, TIERS
 
# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
 
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
 
TIER_COLOURS = {"Easy": "#86b6ef", "Medium": "#2a78d6", "Hard": "#104281"}
SCORE_COLOURS = ["#86b6ef", "#2a78d6", "#104281"]
 
SITE_LABELS = {"shopping": "Shopping", "shopping_admin": "Shopping admin", "reddit": "Reddit"}
 
DIMENSION_LABELS = {"pages_to_traverse": "Pages to traverse", "retrieval_type": "Retrieval type", "interaction": "Interaction", "target_locatability": "Target locatability",}
 
DPI = 300
# surface gap between touching fills
GAP_PT = 2.0
# rounded data-end
ROUND_PT = 4.0
 
 
def _style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Segoe UI", "Helvetica", "Arial"],
        "text.color": INK,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK_SECONDARY,
        "xtick.color": INK_MUTED,
        "ytick.color": INK_MUTED,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.bottom": False,
        "xtick.bottom": False,
        "ytick.left": False,
        "axes.grid": False,
        "figure.dpi": DPI,
    })
 