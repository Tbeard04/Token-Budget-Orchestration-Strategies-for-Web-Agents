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

# Layout helpers
# Create a new figure and adjust the layout
def _canvas(w: float, h: float, left: float, right: float, top: float, bottom: float):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    return fig, ax
 
# Add a title and subtitle to the figure
def _header(fig, title: str, subtitle: str | None, x: float = 0.015) -> None:
    # one point, in figure fraction
    pt = 1.0 / (fig.get_figheight() * 72.0)
    fig.text(x, 1 - 14 * pt, title, fontsize=12.5, fontweight="semibold", color=INK, va="top", ha="left")
    if subtitle:
        fig.text(x, 1 - 34 * pt, subtitle, fontsize=9.2, color=INK_SECONDARY, va="top", ha="left")
 
# Add a legend to the figure
def _legend(fig, labels: list[str], colours: list[str], x: float = 0.015, y: float = 0.012) -> None:
    handles = [Patch(facecolor=c, edgecolor="none", label=l)
               for l, c in zip(labels, colours)]
    leg = fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(x, y), ncol=len(labels), frameon=False, fontsize=8.8,
                     handlelength=0.85, handleheight=0.85, columnspacing=1.8,
                     handletextpad=0.55)
    for t in leg.get_texts():
        t.set_color(INK_SECONDARY)
 
# Save the figure to a file
def _save(fig, out: Path) -> None:
    fig.savefig(out, dpi=DPI, bbox_inches="tight", pad_inches=0.26)
    plt.close(fig)


# Helper functions
# Calculate the luminance of a hex colour
def _luminance(hex_colour: str) -> float:
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
 
# Determine the ink or white inside a coloured fill, whichever clears contrast
def _on_fill(hex_colour: str) -> str:
    """Ink or white inside a coloured fill, whichever clears contrast."""
    return "#ffffff" if _luminance(hex_colour) < 0.42 else INK
 
# Convert points to data coordinates
def _pt_to_data(ax, pts: float) -> tuple[float, float]:
    inv = ax.transData.inverted()
    scale = ax.figure.dpi / 72.0
    x0, y0 = inv.transform((0, 0))
    x1, y1 = inv.transform((pts * scale, pts * scale))
    return abs(x1 - x0), abs(y1 - y0)

