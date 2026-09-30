from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import math


#Styling
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11
})
#colours for the different strategies
COLOURS = {
    "A": "#2196F3", #blue
    "B": "#FF9800", #orange
    "C": "#4CAF50" #green
}

# Terminations caused by running out of budget rather than by agent behaviour
BUDGET_TERMINATIONS = {"safety_token_cap", "budget_would_exceed", "budget_exhausted_mid_step"}
#terminations caused by the agent getting stuck -- these are Stop examples
STUCK_TERMINATIONS = {"navigation_cycle", "repeated_action_failure"}

#Strategy C only
#the router chose to stop (its own category, never a failure mode of the agent)
ROUTER_TERMINATIONS = {"router_stop"}
ROUTER_COLOUR = "#4a3aa7" #violet

#function to collapse a termination_reason into the five groups used in every figure
def outcome_group(reason: str) -> str:
    if reason == "success":
        return "success"
    if reason in ROUTER_TERMINATIONS:
        return "router stop"
    if reason in BUDGET_TERMINATIONS:
        return "budget exhausted"
    if reason in STUCK_TERMINATIONS:
        return "stuck (guards)"
    if reason == "env_terminated":
        return "wrong answer"
    return "other"

#order and colours for the different outcome groups
OUTCOME_ORDER = ["success", "wrong answer", "budget exhausted", "stuck (guards)", "router stop", "other"]
OUTCOME_COLOURS = {"success": "#1baf7a", "wrong answer": "#9E9E9E", "budget exhausted": "#eb6834", "stuck (guards)": "#e34948", "router stop": ROUTER_COLOUR, "other": "#c9c8c2"}


#95% Wilson interval for k successes out of n (stays inside 0-1 at small n)
#used for confidence intervals in the cost curve plot
#wilson puts error bars on a success rate
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, centre - half), min(1.0, centre + half)

#function for the exact two-sided McNemar test on the discordant pairs (x wins, y wins)
#mcnemar_exact tests whether one strategy beats another or just got lucky
def mcnemar_exact(x_only: int, y_only: int) -> float:
    n = x_only + y_only
    if n == 0:
        return 1.0
    k = min(x_only, y_only)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")

# Loading
def load(path: str, clean: bool = True) -> pd.DataFrame:
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
    if "use_for_reward" in df.columns:
        n = len(df)
        if clean:
            df = df[df["use_for_reward"] == True].copy()
            print(f"[load] {Path(path).name}: {len(df)}/{n} episodes kept "
                  f"(decontaminated: {n - len(df)} excluded by use_for_reward)")
        else:
            print(f"[load] {Path(path).name}: all {n} episodes (contaminated rows INCLUDED)")
    return df


def join_tiers(df: pd.DataFrame, tiers_path: str) -> pd.DataFrame:
    if not tiers_path or not Path(tiers_path).exists():
        return df
    tiers = pd.read_json(tiers_path, lines=True)
    #final_episodes files already carry the labels; merging again would create _x/_y columns
    cols = [c for c in ["task_id", "difficulty_tier", "task_category", "rubric_total"]
            if c in tiers.columns and (c == "task_id" or c not in df.columns)]
    if cols == ["task_id"]:
        print("Difficulty tiers already present in the episodes (--tiers not needed)")
        return df
    df = df.merge(tiers[cols], on="task_id", how="left")
    matched = df["difficulty_tier"].notna().sum()
    print(f"Joined difficulty tiers: {matched}/{len(df)} episodes matched")
    return df


# Shared tables
def success_by_budget(df: pd.DataFrame) -> pd.DataFrame:
    print_section("Success Rate by Budget Level (Cost Curve)")

    rows = []
    for budget, grp in df.groupby("budget_level"):
        distinct_solved = grp.loc[grp["success"] == True, "task_id"].nunique()
        distinct_total = grp["task_id"].nunique()
        rows.append({
            "budget_level": budget,
            "episodes": len(grp),
            "successes": int(grp["success"].sum()),
            "episode_SR": grp["success"].mean(),
            "tasks_solved": distinct_solved,
            "tasks_total": distinct_total,
            "task_SR": distinct_solved / distinct_total if distinct_total else 0,
            "median_tokens": grp["total_tokens"].median(),
            "median_steps": grp["steps"].median(),
        })

    tbl = pd.DataFrame(rows).set_index("budget_level")
    print(tbl.to_string(float_format=lambda x: f"{x:.2%}" if x < 1 else f"{x:.0f}"))

    any_success = df.loc[df["success"] == True, "task_id"].nunique()
    total_tasks = df["task_id"].nunique()
    print(f"\n Distinct tasks solved at ANY budget: {any_success}/{total_tasks}"
          f" ({any_success/total_tasks:.0%})")
    return tbl

#table for success rate by site
def success_by_site(df: pd.DataFrame) -> pd.DataFrame:
    print_section("Success Rate by Site")
    rows = []
    for site, grp in df.groupby("site"):
        distinct_solved = grp.loc[grp["success"] == True, "task_id"].nunique()
        distinct_total = grp["task_id"].nunique()
        rows.append({
            "site": site,
            "episodes": len(grp),
            "episode_SR": grp["success"].mean(),
            "tasks_solved": distinct_solved,
            "tasks_total": distinct_total,
            "task_SR": distinct_solved / distinct_total if distinct_total else 0,
            "median_tokens": grp["total_tokens"].median(),
            "median_steps": grp["steps"].median(),
        })
    tbl = pd.DataFrame(rows).set_index("site")
    print(tbl.to_string(float_format=lambda x: f"{x:.2%}" if x < 1 else f"{x:.0f}"))
    return tbl

#table for task solvability
def task_solvability(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    print_section("Task Solvability")

    solved = df[df["success"] == True].groupby("task_id").agg(
        site=("site", "first"),
        min_budget=("budget_level", "min"),
        max_budget=("budget_level", "max"),
        times_solved=("success", "sum"),
        budgets_tested=("budget_level", "nunique"),
    ).sort_values("min_budget")

    total_tasks = df["task_id"].nunique()
    print(f"{len(solved)}/{total_tasks} tasks solved at least once "
          f"({len(solved)/total_tasks:.0%})\n")

    if len(solved):
        if verbose:
            print(solved.to_string())
        else:
            # Summarise by minimum budget rather than listing every task
            by_min = solved.groupby("min_budget").size()
            print(" Tasks first solved at each budget level:")
            for budget, n in by_min.items():
                print(f"     {budget:>6}: {n:>3} tasks")
        print(f"\n Minimum budget needed (median of solved tasks): "
              f"{solved['min_budget'].median():.0f}")
    else:
        print(" No tasks solved at any budget level.")

    return solved

#table for termination reasons
def termination_reasons(df: pd.DataFrame) -> pd.Series:
    print_section("Termination Reasons")
    counts = df["termination_reason"].value_counts()
    total = len(df)
    for reason, n in counts.items():
        print(f"   {reason:30s} {n:5d}  ({n/total:.0%})")

    budget_driven = df["termination_reason"].isin(BUDGET_TERMINATIONS).sum()
    stuck = df["termination_reason"].isin(STUCK_TERMINATIONS).sum()
    print(f"\n budget-driven total: {budget_driven:5d}  ({budget_driven/total:.0%})")
    print(f"stuck (guard-fired): {stuck:5d}  ({stuck/total:.0%})")
    return counts

#table for token distribution
def token_distribution(df: pd.DataFrame) -> None:
    print_section("Token Distribution")
    toks = df["total_tokens"].dropna()
    print(f"episodes: {len(toks)}")
    print(f"min: {toks.min():.0f}")
    print(f"25th percentile: {toks.quantile(0.25):.0f}")
    print(f"median: {toks.median():.0f}")
    print(f"75th percentile: {toks.quantile(0.75):.0f}")
    print(f"max: {toks.max():.0f}")
    succ = df.loc[df["success"] == True, "total_tokens"]
    if len(succ):
        print(f" median of successes: {succ.median():.0f}  (n={len(succ)})")

#table for per-step token cost
def per_step_cost(df: pd.DataFrame) -> None:
    print_section("Per-step Token Cost")
    df_valid = df[df["steps"] > 0].copy()
    if df_valid.empty:
        print("no episodes with steps > 0")
        return
    df_valid["tok_per_step"] = df_valid["total_tokens"] / df_valid["steps"]
    by_site = df_valid.groupby("site")["tok_per_step"].agg(["median", "mean", "std"])
    print(by_site.to_string())
    overall = df_valid["tok_per_step"]
    print(f"\n overall median: {overall.median():.0f}  "
          f"mean: {overall.mean():.0f}  std: {overall.std():.0f}")

#table for success rate by difficulty tier
def difficulty_breakdown(df: pd.DataFrame) -> None:
    if "difficulty_tier" not in df.columns:
        return
    print_section("Success Rate by Difficulty Tier (RQ2)")
    tier_order = ["Easy", "Medium", "Hard"]
    df = df.copy()
    df["difficulty_tier"] = pd.Categorical(df["difficulty_tier"], categories=tier_order, ordered=True)

    rows = []
    for (tier, budget), grp in df.groupby(["difficulty_tier", "budget_level"],observed=True):
        distinct_solved = grp.loc[grp["success"] == True, "task_id"].nunique()
        distinct_total = grp["task_id"].nunique()
        rows.append({
            "tier": tier,
            "budget": budget,
            "episodes": len(grp),
            "episode_SR": grp["success"].mean(),
            "tasks_solved": distinct_solved,
            "tasks_total": distinct_total,
            "task_SR": distinct_solved / distinct_total if distinct_total else 0,
        })
    tbl = pd.DataFrame(rows).set_index(["tier", "budget"])
    print(tbl.to_string(float_format=lambda x: f"{x:.2%}" if x < 1 else f"{x:.0f}"))

#Shared plots
def _strategy_of(df: pd.DataFrame) -> str:
    return df["strategy"].iloc[0] if "strategy" in df.columns else "?"


def plot_cost_curve(df: pd.DataFrame, out_dir: Path) -> None:
    strategy = _strategy_of(df)
    colour = COLOURS.get(strategy, "#333")

    rows = []
    for budget, grp in df.groupby("budget_level"):
        solved = grp.loc[grp["success"] == True, "task_id"].nunique()
        total = grp["task_id"].nunique()
        rows.append({
            "budget": budget,
            "episode_SR": grp["success"].mean(),
            "task_SR": solved / total if total else 0,
        })
    tbl = pd.DataFrame(rows)

    fig, ax = plt.subplots()
    ax.plot(tbl["budget"], tbl["episode_SR"], "o-", color=colour, linewidth=2, markersize=8, label="Episode SR")
    ax.plot(tbl["budget"], tbl["task_SR"], "s--", color=colour, linewidth=1.5, markersize=7, alpha=0.6, label="Task SR (distinct)")
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Strategy {strategy}: Success Rate vs Token Budget")
    ax.set_ylim(-0.02, max(tbl[["episode_SR", "task_SR"]].max().max() * 1.3, 0.25))
    ax.set_xticks(tbl["budget"])
    ax.set_xticklabels([f"{x//1000}k" for x in tbl["budget"]])
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"cost_curve_{strategy.lower()}.png"
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")
    plt.close()


def plot_termination_reasons(df: pd.DataFrame, out_dir: Path) -> None:
    strategy = _strategy_of(df)
    counts = df["termination_reason"].value_counts()

    fig, ax = plt.subplots()
    colours = []
    for reason in counts.index:
        if reason == "success":
            colours.append("#4CAF50")
        elif reason in BUDGET_TERMINATIONS:
            colours.append("#FF9800")
        elif reason in STUCK_TERMINATIONS:
            colours.append("#F44336")
        else:
            colours.append("#9E9E9E")
    ax.barh(counts.index, counts.values, color=colours)
    ax.set_xlabel("Episode Count")
    ax.set_title(f"Strategy {strategy}: Termination Reasons")
    ax.invert_yaxis()
    fig.tight_layout()
    path = out_dir / f"termination_reasons_{strategy.lower()}.png"
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")
    plt.close()


def plot_tokens_by_budget(df: pd.DataFrame, out_dir: Path) -> None:
    strategy = _strategy_of(df)
    budgets = sorted(df["budget_level"].unique())
    data = [df.loc[df["budget_level"] == b, "total_tokens"].values for b in budgets]

    fig, ax = plt.subplots()
    bp = ax.boxplot(data, tick_labels=[f"{b//1000}k" for b in budgets], patch_artist=True)
    colour = COLOURS.get(strategy, "#333")
    for patch in bp["boxes"]:
        patch.set_facecolor(colour)
        patch.set_alpha(0.4)
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Actual Tokens Consumed")
    ax.set_title(f"Strategy {strategy}: Token Consumption by Budget Level")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"token_boxplot_{strategy.lower()}.png"
    fig.savefig(path, dpi=150)
    print(f" saved: {path}")
    plt.close()


def plot_success_by_site(df: pd.DataFrame, out_dir: Path) -> None:
    strategy = _strategy_of(df)
    pivot = df.pivot_table(values="success", index="budget_level", columns="site", aggfunc="mean")
    fig, ax = plt.subplots()
    pivot.plot(kind="bar", ax=ax, width=0.7)
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Strategy {strategy}: Success Rate by Site and Budget")
    ax.set_xticklabels([f"{x//1000}k" for x in pivot.index], rotation=0)
    ax.legend(title="Site")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"success_by_site_{strategy.lower()}.png"
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")
    plt.close()


def plot_difficulty_curve(df: pd.DataFrame, out_dir: Path) -> None:
    if "difficulty_tier" not in df.columns:
        return
    strategy = _strategy_of(df)
    tier_colours = {"Easy": "#4CAF50", "Medium": "#FF9800", "Hard": "#F44336"}

    fig, ax = plt.subplots()
    plotted = False
    for tier in ["Easy", "Medium", "Hard"]:
        sub = df[df["difficulty_tier"] == tier]
        if sub.empty:
            continue
        tbl = sub.groupby("budget_level")["success"].mean()
        ax.plot(tbl.index, tbl.values, "o-", label=tier,
                color=tier_colours[tier], linewidth=2, markersize=7)
        plotted = True
    if not plotted:
        plt.close()
        return
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Strategy {strategy}: Success by Difficulty Tier (RQ2)")
    ax.set_xticks(sorted(df["budget_level"].unique()))
    ax.set_xticklabels([f"{x//1000}k" for x in sorted(df["budget_level"].unique())])
    ax.legend(title="Tier")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"difficulty_curve_{strategy.lower()}.png"
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")
    plt.close()


def run_shared_analysis(df: pd.DataFrame, out_dir: Path, verbose_tasks: bool = False) -> None:
    success_by_budget(df)
    success_by_site(df)
    task_solvability(df, verbose=verbose_tasks)
    termination_reasons(df)
    token_distribution(df)
    per_step_cost(df)
    #stop_signal_analysis(df)
    difficulty_breakdown(df)

    print_section("Plots")
    plot_cost_curve(df, out_dir)
    plot_termination_reasons(df, out_dir)
    plot_tokens_by_budget(df, out_dir)
    plot_success_by_site(df, out_dir)
    plot_difficulty_curve(df, out_dir)