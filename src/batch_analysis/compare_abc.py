"""
analysis/compare_ab.py - Strategy A vs Strategy B vs Strategy C comparison
"""
from __future__ import annotations

import argparse
import statistics
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from batch_analysis.shared import (load, join_tiers, print_section, wilson, mcnemar_exact, outcome_group,COLOURS, MARKERS, NAMES, TIER_ORDER, OUTCOME_ORDER, OUTCOME_COLOURS)

#pairs of strategies to compare
PAIRS = [("A", "B"), ("A", "C"), ("B", "C")]

#"A vs B" or "A vs B vs C" depending on what was loaded
def vs(d: dict) -> str:
    return " vs ".join(d)


#the strategy pairs that were loaded
def present_pairs(d: dict) -> list[tuple[str, str]]:
    return [(x, y) for x, y in PAIRS if x in d and y in d]

#function to label the budgets
def budget_labels(budgets) -> list[str]:
    return [f"{b // 1000}k" for b in budgets]

#table for success rate by budget
def comparison_table(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print_section(f"{vs(d)}: Success Rate by Budget")

    budgets = sorted(set().union(*[set(df["budget_level"]) for df in d.values()]))
    rows = []
    for budget in budgets:
        row = {"budget": budget}
        for s, df in d.items():
            g = df[df["budget_level"] == budget]
            row[f"{s}_n"] = len(g)
            row[f"{s}_SR"] = g["success"].mean() if len(g) else float("nan")
            row[f"{s}_med_tok"] = g["total_tokens"].median() if len(g) else float("nan")
        rows.append(row)
    tbl = pd.DataFrame(rows).set_index("budget")
    #success-rate gap for each pair, second minus first (positive = second is better)
    for x, y in present_pairs(d):
        tbl[f"{y}-{x}_SR"] = tbl[f"{y}_SR"] - tbl[f"{x}_SR"]

    with pd.option_context("display.float_format", lambda x: f"{x:.3f}", "display.width", 200):
        print(tbl.to_string())
    return tbl


def equivalent_budget(d: dict[str, pd.DataFrame]) -> None:
    if "A" not in d:
        return
    print_section(f"{vs(d)}: Equivalent Budget (relative to A)")

    a_sr = d["A"].groupby("budget_level")["success"].mean()
    for s, df in d.items():
        if s == "A":
            continue
        print(f"\n Strategy {s}")
        for s_budget, s_val in df.groupby("budget_level")["success"].mean().items():
            closest = (a_sr - s_val).abs().idxmin()
            ratio = s_budget / closest if closest else float("inf")
            print(f" {s} @ {s_budget//1000:>2}k ({s_val:>5.1%})  "
                  f"~=  A @ {closest//1000:>2}k ({a_sr[closest]:>5.1%})   "
                  f"[{s} needs {ratio:.1f}x the budget]")

#table for cost ratio on shared successes
def cost_ratio(d: dict[str, pd.DataFrame]) -> None:
    if "A" not in d:
        return
    print_section(f"{vs(d)}: Cost Ratio on Shared Successes (relative to A)")

    a = d["A"]
    a_solved = set(a.loc[a["success"] == True, "task_id"])
    for s, df in d.items():
        if s == "A":
            continue
        s_solved = set(df.loc[df["success"] == True, "task_id"])
        ratios = []
        for tid in a_solved & s_solved:
            #cheapest success of each strategy on this task
            a_cost = a.loc[(a["task_id"] == tid) & (a["success"] == True), "total_tokens"].min()
            s_cost = df.loc[(df["task_id"] == tid) & (df["success"] == True), "total_tokens"].min()
            if a_cost and a_cost > 0:
                ratios.append(s_cost / a_cost)
        print(f"\n Strategy {s}")
        if not ratios:
            print("no shared successes to compare")
            continue
        print(f"tasks solved by both: {len(ratios)}")
        print(f"median {s}/A cost ratio: {statistics.median(ratios):.2f}x")
        print(f"mean {s}/A cost ratio: {statistics.mean(ratios):.2f}x")
        print(f"range: {min(ratios):.2f}x - {max(ratios):.2f}x")


#same task, same budget, both strategies kept after decontamination ---> one pair
#exact McNemar on the pairs where only one of the two succeeded
def paired_tests(d: dict[str, pd.DataFrame]) -> None:
    print_section(f"{vs(d)}: Paired Comparison (same task, same budget), exact McNemar")

    key = ["task_id", "budget_level"]
    for x, y in present_pairs(d):
        m = d[x][key + ["success"]].merge(d[y][key + ["success"]], on=key, suffixes=(f"_{x}", f"_{y}"))
        rows = []
        groups = [(b, g) for b, g in m.groupby("budget_level")] + [("all", m)]
        for budget, g in groups:
            sx = g[f"success_{x}"].astype(bool)
            sy = g[f"success_{y}"].astype(bool)
            x_only = int((sx & ~sy).sum())
            y_only = int((~sx & sy).sum())
            rows.append({
                "budget": budget if budget == "all" else f"{budget // 1000}k",
                "pairs": len(g),
                f"{x}_SR": round(sx.mean(), 3),
                f"{y}_SR": round(sy.mean(), 3),
                "both": int((sx & sy).sum()),
                f"{x}_only": x_only,
                f"{y}_only": y_only,
                "p": round(mcnemar_exact(x_only, y_only), 4),
            })
        print(f" {x} vs {y}")
        print(pd.DataFrame(rows).set_index("budget").to_string())
        print()

#differences in task-level performance
def task_level_comparison(d: dict[str, pd.DataFrame]) -> None:
    print_section(f"{vs(d)}: Task-Level Comparison (solved at any budget)")

    names = list(d)
    solved = {s: set(df.loc[df["success"] == True, "task_id"]) for s, df in d.items()}
    all_tasks = set().union(*[set(df["task_id"]) for df in d.values()])

    #every solved / not-solved pattern across the loaded strategies
    for pattern in itertools.product([True, False], repeat=len(names)):
        tasks = {t for t in all_tasks
                 if all((t in solved[s]) == want for s, want in zip(names, pattern))}
        winners = [s for s, want in zip(names, pattern) if want]
        label = ("neither" if len(names) == 2 else "none") if not winners else \
            (" + ".join(winners) + (" only" if len(winners) < len(names) else " (all)"))
        print(f"solved by {label:14s} {len(tasks):>4}")

    #B's wins over A: did the Critic contribute?
    if "A" in d and "B" in d:
        a, b = d["A"], d["B"]
        b_only = solved["B"] - solved["A"]
        if b_only:
            print(f"\n Tasks B solved that A could not:")
            with_rev = 0
            for tid in sorted(b_only):
                ep = b[(b["task_id"] == tid) & (b["success"] == True)]
                ep = ep.loc[ep["budget_level"].idxmin()]
                revs = int(ep.get("critic_revisions", 0) or 0)
                with_rev += bool(revs)
                a_eps = a[a["task_id"] == tid]
                a_reason = a_eps.loc[a_eps["budget_level"].idxmax(), "termination_reason"] if len(a_eps) else "not run"
                print(f"task {tid:>4} @ {ep['budget_level']:>6}: {ep['steps']} steps, "
                      f"{revs} revisions   (A failed: {a_reason})")
            print(f"\n {with_rev}/{len(b_only)} involved a Critic revision")
            print(f"{len(b_only) - with_rev}/{len(b_only)} succeeded with the Planner alone")

    #did the Critic contribute to B's wins over A
    if "A" in d and "C" in d:
        c = d["C"]
        lost = solved["A"] - solved["C"]
        gained = solved["C"] - solved["A"]
        print(f"\n Tasks A solved that C never did: {len(lost)}")
        if lost:
            why = c[c["task_id"].isin(lost)]["termination_reason"].map(outcome_group).value_counts()
            print("C's episodes on those tasks ended as: " + ", ".join(f"{k} {v}" for k, v in why.items()))
        print(f"Tasks C solved that A never did: {len(gained)}"
              + (f"{sorted(gained)[:15]}" if gained else ""))

#failure mode shift analysis for each strategy
def failure_mode_shift(d: dict[str, pd.DataFrame]) -> None:
    print_section(f"{vs(d)}: Failure Mode Shift")

    for s, df in d.items():
        fails = df[df["success"] == False]
        if fails.empty:
            continue
        groups = fails["termination_reason"].map(outcome_group).value_counts()
        n = len(fails)
        print(f" Strategy {s}: {n} failures")
        for g in OUTCOME_ORDER[1:]:
            if groups.get(g, 0):
                print(f"{g:18s} {groups[g]:>5}  ({groups[g] / n:.0%})")
        print()


# Plots for the comparison
#helper function to save the figures
def _finish(fig, ax, out_dir: Path, name: str, legend: bool = True) -> None:
    if legend:
        ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")
    plt.close(fig)

#plot the cost vs success rate curve
def plot_cost_curves_overlay(d: dict, out_dir: Path, tag: str) -> None:
    fig, ax = plt.subplots()
    budgets = sorted(set().union(*[set(df["budget_level"]) for df in d.values()]))
    for s, df in d.items():
        g = df.groupby("budget_level")["success"].agg(["sum", "count"])
        sr = g["sum"] / g["count"]
        lo, hi = zip(*[wilson(int(k), int(n)) for k, n in zip(g["sum"], g["count"])])
        ax.fill_between(g.index, lo, hi, color=COLOURS[s], alpha=0.12, linewidth=0)
        ax.plot(g.index, sr, MARKERS[s], color=COLOURS[s], linewidth=2, markersize=8, label=NAMES[s])
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Success Rate (shaded: 95% interval)")
    ax.set_title(f"Success Rate vs Token Budget: {vs(d)}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels(budget_labels(budgets))
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    _finish(fig, ax, out_dir, f"cost_curve_{tag}.png")

#plot the cost vs performance curve
def plot_cost_performance(d: dict, out_dir: Path, tag: str) -> None:
    fig, ax = plt.subplots()
    notes = []
    for s, df in d.items():
        g = df.groupby("budget_level").agg(sr=("success", "mean"), tok=("total_tokens", "mean"))
        zero = g[g["tok"] <= 0]
        if len(zero):
            notes.append(f"{s} at " + ", ".join(f"{b // 1000}k" for b in zero.index) + ": no tokens spent (cannot afford a single step), 0% success")
        g = g[g["tok"] > 0]
        ax.plot(g["tok"], g["sr"], MARKERS[s], color=COLOURS[s], linewidth=2, markersize=8, label=NAMES[s])
        #label the budget beside each point, in muted text
        for budget, row in g.iterrows():
            ax.annotate(f"{budget // 1000}k", (row["tok"], row["sr"]), textcoords="offset points", xytext=(6, -12), fontsize=8, color="#52514e")
    ax.set_xscale("log")
    ticks = [t for t in [1000, 2000, 4000, 8000, 16000, 32000, 64000] if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1000}k" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Mean Tokens Actually Spent per Episode (point labels: budget)")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Cost vs Performance: {vs(d)}")
    ax.set_ylim(bottom=0)
    if notes:
        ax.text(0.99, 0.02, "\n".join(notes), transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#52514e")
    _finish(fig, ax, out_dir, f"cost_performance_{tag}.png")

def plot_per_step_cost_comparison(d: dict, out_dir: Path, tag: str) -> None:
    data, labels = [], []
    for s, df in d.items():
        v = df[df["steps"] > 0]
        data.append((v["total_tokens"] / v["steps"]).values)
        labels.append(f"Strategy {s}")

    fig, ax = plt.subplots()
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, s in zip(bp["boxes"], d):
        patch.set_facecolor(COLOURS[s])
        patch.set_alpha(0.5)
    ax.set_ylabel("Tokens per Step")
    ax.set_title(f"Per-Step Token Cost: {vs(d)}")
    _finish(fig, ax, out_dir, f"per_step_cost_{tag}.png", legend=False)


def plot_token_efficiency(d: dict, out_dir: Path, tag: str) -> None:
    fig, ax = plt.subplots()
    budgets = sorted(set().union(*[set(df["budget_level"]) for df in d.values()]))
    for s, df in d.items():
        g = df.groupby("budget_level").agg(sr=("success", "mean"), tok=("total_tokens", "mean"))
        eff = (g["sr"] / g["tok"] * 1000).where(g["tok"] > 0, 0)
        ax.plot(g.index, eff, MARKERS[s], color=COLOURS[s], linewidth=2, markersize=8, label=NAMES[s])
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Token Efficiency (SR per 1k tokens)")
    ax.set_title(f"Token Efficiency by Budget Level: {vs(d)}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels(budget_labels(budgets))
    ax.minorticks_off()
    _finish(fig, ax, out_dir, f"token_efficiency_{tag}.png")


def plot_outcome_mix(d: dict, out_dir: Path, tag: str) -> None:
    shares = {}
    for s, df in d.items():
        counts = df["termination_reason"].map(outcome_group).value_counts()
        shares[s] = counts / counts.sum()
    groups = [g for g in OUTCOME_ORDER if any(shares[s].get(g, 0) for s in d)]

    fig, ax = plt.subplots(figsize=(10, 1.4 + 0.9 * len(d)))
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for g in groups:
        vals = np.array([shares[s].get(g, 0) for s in d])
        ax.barh(y, vals, left=left, color=OUTCOME_COLOURS[g], edgecolor="white", linewidth=2, label=g)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v >= 0.06:
                ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=9, color="white")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([f"Strategy {s}" for s in d])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of Episodes")
    ax.set_title(f"How Episodes End: {vs(d)}")
    ax.legend(ncol=len(groups), loc="upper center", bbox_to_anchor=(0.5, -0.35 if len(d) > 2 else -0.45), frameon=False, fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    path = out_dir / f"outcome_mix_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved: {path}")
    plt.close(fig)


#plot the success by difficulty tier, all budgets pooled, with 95% intervals
def plot_difficulty_comparison(d: dict, out_dir: Path, tag: str) -> None:
    if not all("difficulty_tier" in df.columns for df in d.values()):
        return
    fig, ax = plt.subplots()
    width = 0.8 / len(d)
    x = np.arange(len(TIER_ORDER))
    for i, (s, df) in enumerate(d.items()):
        srs, errs = [], [[], []]
        for tier in TIER_ORDER:
            g = df[df["difficulty_tier"] == tier]["success"]
            k, n = int(g.sum()), len(g)
            sr = k / n if n else 0
            lo, hi = wilson(k, n)
            srs.append(sr)
            errs[0].append(max(0, sr - lo) if n else 0)
            errs[1].append(max(0, hi - sr) if n else 0)
        ax.bar(x + (i - (len(d) - 1) / 2) * width, srs, width * 0.92, color=COLOURS[s], yerr=errs, capsize=3, error_kw={"elinewidth": 1, "ecolor": "#52514e"}, label=NAMES[s])
    ax.set_xticks(x)
    ax.set_xticklabels(TIER_ORDER)
    ax.set_xlabel("Difficulty Tier")
    ax.set_ylabel("Success Rate (all budgets)")
    ax.set_title(f"Success by Difficulty Tier: {vs(d)}")
    _finish(fig, ax, out_dir, f"difficulty_{tag}.png")


# Main function to run the comparison
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--c", default=None)
    ap.add_argument("--tiers", default=None)
    ap.add_argument("--out", default="../data/processed/6_budgets_ALL_tasks_decontaminted_batches/comparisons")
    args = ap.parse_args()

    paths = {"A": args.a, "B": args.b, "C": args.c}
    d = {s: load(p) for s, p in paths.items() if p}
    print("Loaded " + ", ".join(f"{len(df)} {s}" for s, df in d.items()) + " episodes")

    if args.tiers:
        d = {s: join_tiers(df, args.tiers) for s, df in d.items()}

    #restrict to tasks present in every loaded strategy, so the comparison is like for like
    shared = set.intersection(*[set(df["task_id"]) for df in d.values()])
    if len(shared) < max(df["task_id"].nunique() for df in d.values()):
        print(f"\n[compare] restricting to {len(shared)} tasks present in ALL datasets (" + ", ".join(f"{s} has {df['task_id'].nunique()}" for s, df in d.items()) + ")")
        d = {s: df[df["task_id"].isin(shared)] for s, df in d.items()}

    comparison_table(d)
    paired_tests(d)
    equivalent_budget(d)
    cost_ratio(d)
    task_level_comparison(d)
    failure_mode_shift(d)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_vs_".join(s.lower() for s in d)

    print_section("Comparison Plots")
    plot_cost_curves_overlay(d, out_dir, tag)
    plot_cost_performance(d, out_dir, tag)
    plot_per_step_cost_comparison(d, out_dir, tag)
    plot_token_efficiency(d, out_dir, tag)
    plot_outcome_mix(d, out_dir, tag)
    plot_difficulty_comparison(d, out_dir, tag)


if __name__ == "__main__":
    main()