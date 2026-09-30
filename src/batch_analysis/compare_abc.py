"""
analysis/compare_ab.py - Strategy A vs Strategy B vs Strategy C comparison
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

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


def task_level_comparison(a: pd.DataFrame, b: pd.DataFrame) -> None:
    print_section("A vs B: Task-Level Comparison")

    all_tasks = set(a["task_id"]) | set(b["task_id"])
    a_solved = set(a.loc[a["success"] == True, "task_id"])
    b_solved = set(b.loc[b["success"] == True, "task_id"])

    both = a_solved & b_solved
    a_only = a_solved - b_solved
    b_only = b_solved - a_solved
    neither = all_tasks - a_solved - b_solved

    print(f"solved by both: {len(both)}")
    print(f"solved by A only: {len(a_only)}")
    print(f"solved by B only: {len(b_only)}")
    print(f"solved by neither: {len(neither)}")

    if b_only:
        print(f"\nTasks B solved that A could not: ")
        with_rev = 0
        for tid in sorted(b_only):
            eps = b[(b["task_id"] == tid) & (b["success"] == True)]
            if eps.empty:
                continue
            ep = eps.loc[eps["budget_level"].idxmin()]
            revs = int(ep.get("critic_revisions", 0))
            if revs:
                with_rev += 1
            # What did A do on this task at its best budget
            a_eps = a[a["task_id"] == tid]
            a_reason = "not run"
            if len(a_eps):
                a_best = a_eps.loc[a_eps["budget_level"].idxmax()]
                a_reason = a_best["termination_reason"]
            print(f"task {tid:>4} @ {ep['budget_level']:>6}: "
                  f"{ep['steps']} steps, {revs} revisions   "
                  f"(A failed: {a_reason})")

        print(f"\n {with_rev}/{len(b_only)} involved a Critic revision")
        print(f" {len(b_only)-with_rev}/{len(b_only)} succeeded with the Planner alone")

    if a_only:
        shown = sorted(a_only)[:15]
        print(f"\n Tasks A solved that B could not: {shown}"
              f"{f' ... +{len(a_only)-15} more' if len(a_only) > 15 else ''}")


def failure_mode_shift(a: pd.DataFrame, b: pd.DataFrame) -> None:
    print_section("A vs B: Failure Mode Shift")

    def profile(df, label):
        fails = df[df["success"] == False]
        if fails.empty:
            return
        budget = fails["termination_reason"].isin(BUDGET_TERMINATIONS).sum()
        stuck = fails["termination_reason"].isin(STUCK_TERMINATIONS).sum()
        wrong = (fails["termination_reason"] == "env_terminated").sum()
        n = len(fails)
        print(f" {label}: {n} failures")
        print(f" budget exhausted: {budget:>5}  ({budget/n:.0%})")
        print(f" stuck (guards): {stuck:>5}  ({stuck/n:.0%})")
        print(f"wrong answer: {wrong:>5}  ({wrong/n:.0%})")

    profile(a, "Strategy A")
    print()
    profile(b, "Strategy B")


def strategy_c_ceiling(a: pd.DataFrame, b: pd.DataFrame) -> None:
    print_section("Strategy C: Theoretical Ceiling")

    a_solved = set(a.loc[a["success"] == True, "task_id"])
    b_solved = set(b.loc[b["success"] == True, "task_id"])
    union = a_solved | b_solved
    all_tasks = set(a["task_id"]) | set(b["task_id"])

    print(f"tasks solved by A: {len(a_solved)}")
    print(f"tasks solved by B: {len(b_solved)}")
    print(f"union (perfect routing): {len(union)}  "
          f"({len(union)/len(all_tasks):.0%} of {len(all_tasks)} tasks)")
    print(f"gain over A alone: +{len(union) - len(a_solved)} tasks")
    print(f"gain over B alone: +{len(union) - len(b_solved)} tasks")

    # Token savings from skipping rubber-stamp Critic calls
    approval_tokens = 0
    for _, row in b.iterrows():
        steps = row.get("step_log")
        if not isinstance(steps, list):
            continue
        for s in steps:
            if s.get("agent_role") == "critic" and not s.get("revised"):
                approval_tokens += (s.get("input_tokens", 0) + s.get("output_tokens", 0))

    b_total = b["total_tokens"].sum()
    if b_total:
        print(f"\nB's total token spend: {b_total:,}")
        print(f"spent on Critic approvals: {approval_tokens:,} "
              f"({approval_tokens/b_total:.0%})")
        print(f"--> recoverable by perfect Critic routing")


# Plots
def plot_cost_curves_overlay(a: pd.DataFrame, b: pd.DataFrame, out_dir: Path) -> None:
    a_tbl = a.groupby("budget_level")["success"].mean()
    b_tbl = b.groupby("budget_level")["success"].mean()

    fig, ax = plt.subplots()
    ax.plot(a_tbl.index, a_tbl.values, "o-", color=COLOURS["A"], linewidth=2, markersize=8, label="Strategy A (single agent)")
    ax.plot(b_tbl.index, b_tbl.values, "s--", color=COLOURS["B"], linewidth=2, markersize=8, label="Strategy B (fixed pipeline)")
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Success Rate")
    ax.set_title("Success Rate vs Token Budget: A vs B")
    budgets = sorted(set(a_tbl.index) | set(b_tbl.index))
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{x//1000}k" for x in budgets])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "cost_curve_a_vs_b.png"
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")
    plt.close()

def plot_per_step_cost_comparison(a: pd.DataFrame, b: pd.DataFrame, out_dir: Path) -> None:
    av = a[a["steps"] > 0].copy()
    bv = b[b["steps"] > 0].copy()
    av["tps"] = av["total_tokens"] / av["steps"]
    bv["tps"] = bv["total_tokens"] / bv["steps"]

    fig, ax = plt.subplots()
    bp = ax.boxplot([av["tps"].values, bv["tps"].values], tick_labels=["Strategy A", "Strategy B"], patch_artist=True, showfliers=False)
    bp["boxes"][0].set_facecolor(COLOURS["A"]); bp["boxes"][0].set_alpha(0.5)
    bp["boxes"][1].set_facecolor(COLOURS["B"]); bp["boxes"][1].set_alpha(0.5)
    ax.set_ylabel("Tokens per Step")
    ax.set_title("Per-Step Token Cost: A vs B")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "per_step_cost_a_vs_b.png"
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")
    plt.close()


def plot_token_efficiency(a: pd.DataFrame, b: pd.DataFrame, out_dir: Path) -> None:
    rows = []
    for label, df, colour in [("A", a, COLOURS["A"]), ("B", b, COLOURS["B"])]:
        for budget, grp in df.groupby("budget_level"):
            mean_tok = grp["total_tokens"].mean()
            eff = grp["success"].mean() / mean_tok * 1000 if mean_tok else 0
            rows.append({"strategy": label, "budget": budget, "eff": eff})
    tbl = pd.DataFrame(rows)

    fig, ax = plt.subplots()
    for label, colour, marker in [("A", COLOURS["A"], "o-"), ("B", COLOURS["B"], "s--")]:
        sub = tbl[tbl["strategy"] == label]
        ax.plot(sub["budget"], sub["eff"], marker, color=colour, linewidth=2, markersize=8, label=f"Strategy {label}")
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Token Efficiency (SR per 1k tokens)")
    ax.set_title("Token Efficiency by Budget Level")
    budgets = sorted(tbl["budget"].unique())
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{x//1000}k" for x in budgets])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / "token_efficiency_a_vs_b.png"
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")
    plt.close()

# Main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="Strategy A JSONL")
    ap.add_argument("--b", required=True, help="Strategy B JSONL")
    ap.add_argument("--tiers", default=None, help="task_metadata.jsonl")
    ap.add_argument("--out", default="../data/processed/6_budgets_ALL_tasks_decontaminted_batches/comparisons")
    args = ap.parse_args()

    a = load(args.a)
    b = load(args.b)
    print(f"Loaded {len(a)} A episodes and {len(b)} B episodes")

    if args.tiers:
        a = join_tiers(a, args.tiers)
        b = join_tiers(b, args.tiers)

    # Restrict to tasks present in both, so the comparison is like for like
    shared = set(a["task_id"]) & set(b["task_id"])
    if len(shared) < max(a["task_id"].nunique(), b["task_id"].nunique()):
        print(f"\n[compare] restricting to {len(shared)} tasks present in BOTH "
              f"datasets (A has {a['task_id'].nunique()}, "
              f"B has {b['task_id'].nunique()})")
        a = a[a["task_id"].isin(shared)]
        b = b[b["task_id"].isin(shared)]

    comparison_table(a, b)
    equivalent_budget(a, b)
    cost_ratio(a, b)
    task_level_comparison(a, b)
    failure_mode_shift(a, b)
    strategy_c_ceiling(a, b)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_section("Comparison Plots")
    plot_cost_curves_overlay(a, b, out_dir)
    plot_per_step_cost_comparison(a, b, out_dir)
    plot_token_efficiency(a, b, out_dir)

    print("\nDone.")

if __name__ == "__main__":
    main()