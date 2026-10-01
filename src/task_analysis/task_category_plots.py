from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

#update the plot parameters
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11
})

#category colours
CATEGORY_COLOURS = {
    "information_retrieval":"#2196F3",
    "navigation":"#03A9F4",
    "other":"#90A4AE",
    "create":"#4CAF50",
    "modify_value":"#FF9800",
    "bulk_action":"#FF5722",
    "delete":"#F44336",
    "purchase":"#9C27B0"
}

#Which categories can be contaminated by an earlier successful run
RISK_GROUPS = {"read-only (no contamination possible)": ["information_retrieval", "navigation", "other",], "state-change (contamination possible)": ["create", "modify_value", "bulk_action", "delete", "purchase",]}

#function to plot the overall categories
def plot_overall(by_category: dict, out_dir: Path) -> None:
    #items = the categories sorted by the number of tasks
    items = sorted(by_category.items(), key=lambda x: x[1])
    #labels = the categories replaced with spaces
    labels = [k.replace("_", " ") for k, _ in items]
    #values = the number of tasks for each category
    values = [v for _, v in items]
    colours = [CATEGORY_COLOURS.get(k, "#90A4AE") for k, _ in items]
    #total = the total number of tasks
    total = sum(values)

    fig, ax = plt.subplots()
    bars = ax.barh(labels, values, color=colours, alpha=0.85)

    #Count and percentage at the end of each bar
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + total * 0.008, bar.get_y() + bar.get_height() / 2,
                f"{v}  ({v/total:.0%})", va="center", fontsize=10)

    ax.set_xlabel("Number of Tasks")
    ax.set_title(f"WebArena Task Categories ({total} single-site tasks)")
    ax.set_xlim(0, max(values) * 1.18)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()

    path = out_dir / "task_categories_overall.png"
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")
    plt.close()

#function to plot the categories by site
def plot_by_site(sites: dict, out_dir: Path) -> None:
    site_names = ["shopping", "shopping_admin", "reddit"]
    #all_cats = the categories sorted by the number of tasks
    all_cats = sorted({c for s in site_names for c in sites.get(s, {}).get("categories", {})},
        key=lambda c: -sum(sites.get(s, {}).get("categories", {}).get(c, {}).get("count", 0) for s in site_names))

    fig, ax = plt.subplots(figsize=(12, 6))
    #n_cats = the number of categories
    n_cats = len(all_cats)
    width = 0.8 / n_cats
    #x = the range of the site names
    x = range(len(site_names))

    #for each category in the categories
    for i, cat in enumerate(all_cats):
        #get the counts for the category
        counts = [sites.get(s, {}).get("categories", {}).get(cat, {}).get("count", 0) for s in site_names]
        #calculate the offset
        offset = (i - n_cats / 2 + 0.5) * width
        #plot the bar
        ax.bar([p + offset for p in x], counts, width, label=cat.replace("_", " "), color=CATEGORY_COLOURS.get(cat, "#90A4AE"), alpha=0.85)

    ax.set_ylabel("Number of Tasks")
    ax.set_title("Task Categories by Site")
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.replace("_", " ") for s in site_names])
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = out_dir / "task_categories_by_site.png"
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")
    plt.close()

#function to plot the risk groups
def plot_risk_groups(by_category: dict, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    #bottom = the bottom of the plot
    bottom = 0
    #total = the total number of tasks
    total = sum(by_category.values())
    #for each group in the risk groups
    for group_label, cats in RISK_GROUPS.items():
        group_total = 0
        #for each category in the categories
        for cat in cats:
            #get the count for the category
            n = by_category.get(cat, 0)
            if not n:
                continue
            #Label inside the segment if there is room
            if n >= total * 0.04:
                ax.text(0, bottom + n / 2, f"{cat.replace('_',' ')}  {n}", ha="center", va="center", fontsize=9, color="white", fontweight="bold")
            bottom += n
            group_total += n

        #Bracket and label the group to the right
        group_start = bottom - group_total
        ax.plot([0.30, 0.34, 0.34, 0.30], [group_start + 2, group_start + 2, bottom - 2, bottom - 2], color="#455A64", linewidth=1.2)
        ax.text(0.37, group_start + group_total / 2,
                f"{group_label}\n{group_total} tasks "
                f"({group_total/total:.0%})",
                va="center", fontsize=10)

    ax.set_ylabel("Number of Tasks")
    ax.set_title("Task Categories by Contamination Risk")
    ax.set_xlim(-0.4, 1.3)
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    path = out_dir / "task_categories_risk.png"
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")
    plt.close()

#function to main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="../../data/processed/diagrams/task_list")
    args = ap.parse_args()

    with open(args.input) as f:
        data = json.load(f)

    by_category = data["summary"]["by_category"]
    sites = data.get("sites", {})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}\nPlots\n{'=' * 60}")
    plot_overall(by_category, out_dir)
    plot_by_site(sites, out_dir)
    plot_risk_groups(by_category, out_dir)

    print(f"\nDone.")

if __name__ == "__main__":
    main()