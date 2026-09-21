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
 
from rubric import TIERS
 
# Colour tokens
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Tier colours
TIER_COLOURS = {"Easy": "#86b6ef", "Medium": "#2a78d6", "Hard": "#104281"}
# Score colours
SCORE_COLOURS = ["#86b6ef", "#2a78d6", "#104281"]
# Site labels
SITE_LABELS = {"shopping": "Shopping", "shopping_admin": "Shopping admin", "reddit": "Reddit"}
# Dimension labels
DIMENSION_LABELS = {"pages_to_traverse": "Pages to traverse", "retrieval_type": "Retrieval type", "interaction": "Interaction", "target_locatability": "Target locatability",}

# DPI
DPI = 300
# Surface gap between touching fills
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
    # Calculate the luminance using the formula
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)
 
# Determine the ink or white inside a coloured fill, whichever clears contrast
def _on_fill(hex_colour: str) -> str:
    return "#ffffff" if _luminance(hex_colour) < 0.42 else INK
 
# Convert points to data coordinates
def _pt_to_data(ax, pts: float) -> tuple[float, float]:
    inv = ax.transData.inverted()
    scale = ax.figure.dpi / 72.0
    x0, y0 = inv.transform((0, 0))
    x1, y1 = inv.transform((pts * scale, pts * scale))
    return abs(x1 - x0), abs(y1 - y0)

def _rounded_hbar(ax, y, width, height, colour) -> None:
    # Horizontal bar: square at the baseline, rounded at the data end.
    if width == 0:
        return
    rx, ry = _pt_to_data(ax, ROUND_PT)
    r_x, r_y = min(rx, abs(width)), min(ry, height / 2)
    yb, yt = y - height / 2, y + height / 2
    verts = [(0, yb), (width - r_x, yb), (width, yb), (width, yb + r_y), (width, yt - r_y), (width, yt), (width - r_x, yt), (0, yt), (0, yb)]
    # Define the path codes
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=colour, edgecolor="none", lw=0, zorder=3))
 

def _fits(ax, text: str, seg_width_data: float, fontsize: float) -> bool:
    # Measure the rendered label before placing it inside a segment.
    ax.figure.canvas.draw()
    t = ax.text(0, 0, text, fontsize=fontsize, alpha=0)
    bb = t.get_window_extent(renderer=ax.figure.canvas.get_renderer())
    t.remove()
    inv = ax.transData.inverted()
    w = abs(inv.transform((bb.width, 0))[0] - inv.transform((0, 0))[0])
    # Return True if the label fits inside the segment, False otherwise
    return w * 1.5 < abs(seg_width_data)
 
 
def _stacked_row(ax, y, parts, total, height, fontsize=8.5) -> None:
    # One 100%-stacked row. parts = [(value, colour), ...]
    gap_x, _ = _pt_to_data(ax, GAP_PT)
    left = 0.0
    for value, colour in parts:
        if value <= 0:
            continue
        w = 100.0 * value / total
        ax.barh(y, max(w - gap_x, 0.01), left=left, height=height, color=colour, edgecolor="none", lw=0, zorder=3)
        if _fits(ax, str(value), w - gap_x, fontsize):
            ax.text(left + (w - gap_x) / 2, y, str(value), ha="center", va="center", fontsize=fontsize, color=_on_fill(colour), zorder=4)
        left += w
 
 
def _pct_axis(ax) -> None:
    # Set the x-axis ticks and labels
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"], fontsize=8.8)

# Create a figure and axis for the tier distribution
def fig_tier_distribution(rows: list, out: Path) -> None:
    # Count the number of tasks in each tier
    counts = Counter(r["difficulty_tier"] for r in rows)
    # Get the number of tasks and the maximum number of tasks in a tier
    n, top = len(rows), max(Counter(r["difficulty_tier"] for r in rows).values())
    # Create a new figure and axis
    fig, ax = _canvas(6.6, 2.8, left=0.115, right=0.985, top=0.72, bottom=0.07)
    # Set the x-axis limits and y-axis limits
    ax.set_xlim(0, top * 1.24)
    ax.set_ylim(-0.66, len(TIERS) - 0.34)
    # Get the y-axis values
    ys = list(range(len(TIERS)))[::-1]
    # Create a bar for each tier
    for y, tier in zip(ys, TIERS):
        c = counts.get(tier, 0)
        _rounded_hbar(ax, y, c, 0.5, TIER_COLOURS[tier])
        # Add the number of tasks in the tier
        ax.text(c + top * 0.025, y, f"{c}", va="center", ha="left",
                fontsize=10.5, color=INK)
        ax.text(c + top * 0.025 + top * 0.085, y, f"{c / n:.0%}", va="center",
                ha="left", fontsize=10.5, color=INK_MUTED)
 
    # Set the y-axis ticks and labels
    ax.set_yticks(ys)
    ax.set_yticklabels(TIERS, fontsize=10.5, color=INK)
    # Set the x-axis ticks to empty
    ax.set_xticks([])
    # Add a header to the figure
    _header(fig, "Task difficulty tiers", None)
    _save(fig, out / "tier_distribution.png")
 
 
 
#Create a figure and axis for the tier by site
def fig_tier_by_site(rows: list, out: Path) -> None:
    # Count the number of tasks in each site
    by_site = defaultdict(Counter)
    for r in rows:
        by_site[r["site"]][r["difficulty_tier"]] += 1
    sites = [s for s in SITE_LABELS if s in by_site] or sorted(by_site)
    # Create a new figure and axis
    fig, ax = _canvas(6.6, 2.9, left=0.175, right=0.905, top=0.72, bottom=0.20)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.66, len(sites) - 0.34)
    ys = list(range(len(sites)))[::-1]
    # Create a stacked row for each site
    for y, site in zip(ys, sites):
        c = by_site[site]
        total = sum(c.values())
        _stacked_row(ax, y, [(c.get(t, 0), TIER_COLOURS[t]) for t in TIERS], total, 0.5)
        ax.text(102, y, f"n={total}", va="center", ha="left", fontsize=9, color=INK_MUTED)
 
    ax.set_yticks(ys)
    ax.set_yticklabels([SITE_LABELS.get(s, s) for s in sites], fontsize=10.5, color=INK)
    _pct_axis(ax)
    _header(fig, "Difficulty composition by site", None)
    _legend(fig, TIERS, [TIER_COLOURS[t] for t in TIERS])
    _save(fig, out / "tier_by_site.png")
 
 
def fig_tier_by_category(rows: list, out: Path) -> None:
    # Count the number of tasks in each category
    by_cat = defaultdict(Counter)
    for r in rows:
        by_cat[r.get("task_category", "unknown")][r["difficulty_tier"]] += 1
    cats = sorted(by_cat, key=lambda c: -sum(by_cat[c].values()))
    #Calculate the height of the figure
    h = 0.40 * len(cats) + 1.7
    #Create a new figure and axis
    fig, ax = _canvas(6.6, h, left=0.245, right=0.895, top=1 - 0.88 / h, bottom=0.62 / h)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.7, len(cats) - 0.3)
    ys = list(range(len(cats)))[::-1]
    #Create a stacked row for each category
    for y, cat in zip(ys, cats):
        c = by_cat[cat]
        total = sum(c.values())
        _stacked_row(ax, y, [(c.get(t, 0), TIER_COLOURS[t]) for t in TIERS], total, 0.5, fontsize=8)
        ax.text(102, y, f"n={total}", va="center", ha="left", fontsize=8.5, color=INK_MUTED)
 
    ax.set_yticks(ys)
    ax.set_yticklabels([c.replace("_", " ") for c in cats], fontsize=9.5, color=INK)
    # Set the x-axis ticks and labels
    _pct_axis(ax)
    # Add a header to the figure
    _header(fig, "Difficulty composition by task category", None)
    _legend(fig, TIERS, [TIER_COLOURS[t] for t in TIERS])
    _save(fig, out / "tier_by_category.png")
 
 
#main
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--metadata", default="../../data/processed/final_annotation_difficulty_tiers/task_metadata.jsonl")
    ap.add_argument("--out", default="../../data/processed/diagrams/difficulty_tiers")
    args = ap.parse_args()
 
    rows = [json.loads(l) for l in Path(args.metadata).read_text().splitlines()
            if l.strip()]
    if not rows:
        raise SystemExit(f"no rows in {args.metadata}")
 
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
 
    _style()
    fig_tier_distribution(rows, out)
    # fig_rubric_total(rows, out)
    fig_tier_by_site(rows, out)
    # fig_dimension_profiles(rows, out)
    fig_tier_by_category(rows, out)
 
 
if __name__ == "__main__":
    main()