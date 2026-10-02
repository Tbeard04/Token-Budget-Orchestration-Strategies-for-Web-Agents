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
#frontier meaning the set of trade offs the router can offer between success and tokens saved
def at_quantile(m: dict, q: float) -> dict:
    return next(r for r in m["frontier"] if r["quantile"] == q)

#get the suggested row
def suggested_row(m: dict) -> dict:
    return next(r for r in m["frontier"] if r["threshold"] == m["suggested_threshold"])

#figure for the frontier
def fig_frontier(models: dict, chosen: int, out: Path) -> None:
    fig, ax = new_fig("Cost–performance frontier of the Strategy C router", (8, 5.6))
    #plot the frontier for each model
    for n, d in models.items():
        m = d["metrics"]
        pts = [(100 * r["tokens_saved_vs_A"], 100 * r["success_kept_vs_A"]) #tokens saved vs always-A and successes kept vs always-A
               for r in m["frontier"] if r["threshold"] is not None] #only the rows with a threshold
        xs, ys = zip(*sorted(pts)) #sort the points by tokens saved vs always-A
        ax.plot(xs, ys, "-", color=MODEL_COLOUR[n], lw=2, alpha=0.9, zorder=3)
        ax.plot(xs, ys, MODEL_MARKER[n], color=MODEL_COLOUR[n], ms=6.5,
                mec=SURFACE, mew=1.5, zorder=4,
                label=f"model {n}  (α = {m['config']['alpha']})")
        s = suggested_row(m)
    #plot the baselines
    b = next(iter(models.values()))["metrics"]["baselines"]
    ref_s, ref_t = b["always_A"]["successes"], b["always_A"]["tokens"] #reference successes and tokens
    for name, mk, dx, dy in (("always_A", "D", -10, 6), ("always_B", "X", 10, 6), ("oracle", "*", -12, -16)):
        x = 100 * (1 - b[name]["tokens"] / ref_t) #tokens saved vs always-A
        y = 100 * b[name]["successes"] / ref_s #successes kept vs always-A
        ax.plot(x, y, mk, color=INK2, ms=10 if mk != "*" else 14, mec=SURFACE, mew=1.2, zorder=6)
        ax.annotate(name.replace("_", "-"), (x, y), xytext=(dx, dy),
                    textcoords="offset points", fontsize=9, color=INK2,
                    ha="right" if dx < 0 else "left")
    #plot the grid
    ax.axhline(100, color=GRID, lw=1, zorder=1)
    ax.axvline(0, color=GRID, lw=1, zorder=1)
    ax.set_xlabel("Tokens saved vs always-A (%)")
    ax.set_ylabel("Successes kept vs always-A (%)")
    ax.set_xlim(-55, 95)
    ax.set_ylim(45, 112)
    ax.legend(loc="lower right", fontsize=9)
    save(fig, out / "frontier.png")

#figure for the mode head: P(cycle) on held-out task–budget pairs
def fig_mode_p_cycle(models: dict, out: Path) -> None:
    fig, ax = new_fig("Mode head: P(cycle) on held-out task–budget pairs")
    bins = np.linspace(0, 1, 41) #bins for the histogram
    for n, d in models.items():
        p = [o["p_cycle"] for o in d["oof"]] #p(cycle) for each model
        ax.hist(p, bins=bins, histtype="step", lw=2, color=MODEL_COLOUR[n],
                label=f"model {n}  (picks cycle {d['metrics']['mode_head']['picks_cycle_frac']:.1%})")
    ax.axvline(0.5, color=INK2, lw=1, ls=(0, (4, 3)))
    ax.text(0.51, ax.get_ylim()[1] * 0.45, "decision\nboundary", fontsize=8.5, color=INK2, va="top")
    ax.set_xlabel("P(cycle)")
    ax.set_ylabel("Task–budget pairs")
    ax.legend(fontsize=9, loc="upper right", bbox_to_anchor=(1.0, 0.98))
    save(fig, out / "mode_p_cycle.png")

#figure for the mode head: P(cycle) by which arm actually won
def fig_mode_by_winner(models: dict, chosen: int, out: Path) -> None:
    fig, ax = new_fig(f"Mode head: P(cycle) by which arm actually won (model {chosen})")
    oof = models[chosen]["oof"] #oof predictions
    groups = {"A only": [], "B only": [], "both": [], "neither": []}
    for o in oof: #group the predictions by which arm actually won
        k = ("A only" if o["a_success"] and not o["b_success"] else
             "B only" if o["b_success"] and not o["a_success"] else
             "both" if o["a_success"] else "neither")
        groups[k].append(o["p_cycle"]) #add the p(cycle) to the group
    names = list(groups)
    bp = ax.boxplot([groups[k] for k in names], #boxplot of the p(cycle) for each group
                    tick_labels=[f"{k}\n(n={len(groups[k])})" for k in names],
                    widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color=INK, lw=1.5),
                    whiskerprops=dict(color=INK2), capprops=dict(color=INK2))
    for patch in bp["boxes"]: #set the color of the boxes
        patch.set(facecolor=MODEL_COLOUR[chosen], alpha=0.35, edgecolor=MODEL_COLOUR[chosen])
    ax.axhline(0.5, color=INK2, lw=1, ls=(0, (4, 3))) #add the decision boundary
    ax.set_ylim(-0.02, 0.56)
    ax.set_ylabel("P(cycle)")
    ax.set_xlabel("Which arm succeeded on the pair")
    auc = models[chosen]["metrics"]["mode_head"]["auc_where_outcomes_differ"] #AUC for the mode head
    ax.text(0.98, 0.88, f"AUC, A-only vs B-only: {auc:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color=INK2)
    save(fig, out / "mode_by_winner.png")

#figure for the stop head: mean P(stop) by step (execute arm, held-out)
def fig_stop_by_step(models: dict, out: Path) -> None:
    fig, ax = new_fig("Stop head: mean P(stop) by step (execute arm, held-out)")
    for n, d in models.items():
        by_step: dict[int, list] = {}
        for o in d["oof"]: #group the predictions by step
            for k, q in enumerate(o["p_stop_A"]):
                by_step.setdefault(k, []).append(q) #add the p(stop) to the group
        ks = [k for k in sorted(by_step) if len(by_step[k]) >= 30]
        ax.plot(ks, [np.mean(by_step[k]) for k in ks], "-", color=MODEL_COLOUR[n], lw=2,
                marker=MODEL_MARKER[n], ms=6, mec=SURFACE, mew=1.2, label=f"model {n}")
    ax.set_xlabel("Step index (website moves so far)")
    ax.set_ylabel("Mean P(stop)")
    ax.set_ylim(0, 1.02)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.legend(fontsize=9, loc="lower right")
    save(fig, out / "stop_by_step.png")

#figure for the stop head: P(stop) at step 0 by budget and difficulty
def fig_stop_step0(models: dict, chosen: int, out: Path) -> None:
    fig, ax = new_fig(f"Stop head: P(stop) at step 0 by budget and difficulty (model {chosen})")
    oof = models[chosen]["oof"]
    width = 0.26
    for i, tier in enumerate(["Easy", "Medium", "Hard"]):
        vals = []
        for b in BUDGETS:
            q = [o["p_stop_A"][0] for o in oof
                 if o["budget_level"] == b and o["difficulty_tier"] == tier and o["p_stop_A"]]
            vals.append(np.mean(q) if q else 0)
        x = np.arange(len(BUDGETS)) + (i - 1) * width
        ax.bar(x, vals, width - 0.03, color=TIER_COLOUR[tier], label=tier, zorder=3)
    ax.set_xticks(np.arange(len(BUDGETS)), [f"{b // 1000}k" for b in BUDGETS])
    ax.set_xlabel("Token budget")
    ax.set_ylabel("Mean P(stop) at step 0")
    ax.set_ylim(0, 1.18)
    ax.legend(title="Difficulty tier", fontsize=9, title_fontsize=9, loc="upper right", ncol=3)
    save(fig, out / "stop_step0.png")

#figure for the stop head: successes and tokens at each budget
def fig_per_budget(models: dict, chosen: int, out: Path) -> None:
    m = models[chosen]["metrics"]
    pb = m["per_budget"]
    x = np.arange(len(BUDGETS))
    w = 0.36
    a_s = [pb[str(b)]["baselines"]["always_A"]["successes"] for b in BUDGETS]
    r_s = [pb[str(b)]["router"]["successes"] for b in BUDGETS]
    a_t = [pb[str(b)]["baselines"]["always_A"]["tokens"] / 1e6 for b in BUDGETS]
    r_t = [pb[str(b)]["router"]["tokens"] / 1e6 for b in BUDGETS]
    #plot the successes and tokens at each budget
    for name, a, r, ylab, title in (
            ("per_budget_successes", a_s, r_s, "Successes",
             f"Successes by budget: model {chosen} vs always-A"),
            ("per_budget_tokens", a_t, r_t, "Tokens spent (millions)",
             f"Tokens spent by budget: model {chosen} vs always-A")):
        fig, ax = new_fig(title)
        ax.bar(x - w / 2, a, w - 0.03, color=INK2, alpha=0.55, label="always-A", zorder=3)
        ax.bar(x + w / 2, r, w - 0.03, color=MODEL_COLOUR[chosen],
               label=f"router (model {chosen})", zorder=3)
        ax.set_xticks(x, [f"{b // 1000}k" for b in BUDGETS])
        ax.set_xlabel("Token budget")
        ax.set_ylabel(ylab)
        ax.legend(fontsize=9, loc="upper left")
        if name == "per_budget_tokens":
            for i in range(len(BUDGETS)):
                if a[i] > 0:
                    ax.text(x[i] + w / 2, r[i] + max(a) * 0.015, f"−{1 - r[i] / a[i]:.0%}", ha="center", va="bottom", fontsize=8.5, color=INK)
        save(fig, out / f"{name}.png")

#main function
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="../data/processed/strategy_c_models")
    ap.add_argument("--out", default="../data/processed/diagrams/strategy_c_model123_plots")
    ap.add_argument("--chosen", type=int, default=2)
    args = ap.parse_args()
 
    models = load(Path(args.models))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    fig_frontier(models, args.chosen, out)
    fig_mode_p_cycle(models, out)
    fig_mode_by_winner(models, args.chosen, out)
    fig_stop_by_step(models, out)
    fig_stop_step0(models, args.chosen, out)
    fig_per_budget(models, args.chosen, out)

if __name__ == "__main__":
    main()
