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


def print_section(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def success_by_budget(df: pd.DataFrame) -> pd.DataFrame:
    """The cost curve: success rate at each budget level.
    
    Reports both episode-level SR and distinct-task-level solvability.
    The pooled 'overall' rate is deliberately NOT shown as it conflates
    budget levels that are different experimental conditions.
    """
    print_section("Success Rate by Budget Level (Cost Curve)")
    
    rows = []
    for budget, grp in df.groupby("budget_level"):
        n_episodes = len(grp)
        n_success = grp["success"].sum()
        sr = grp["success"].mean()
        distinct_solved = grp.loc[grp["success"] == True, "task_id"].nunique()
        distinct_total = grp["task_id"].nunique()
        rows.append({
            "budget_level": budget,
            "episodes": n_episodes,
            "successes": int(n_success),
            "episode_SR": sr,
            "tasks_solved": distinct_solved,
            "tasks_total": distinct_total,
            "task_SR": distinct_solved / distinct_total if distinct_total else 0,
            "median_tokens": grp["total_tokens"].median(),
            "median_steps": grp["steps"].median(),
        })
    
    tbl = pd.DataFrame(rows).set_index("budget_level")
    print(tbl.to_string(float_format=lambda x: f"{x:.2%}" if x < 1 else f"{x:.0f}"))
    
    # Task-level summary
    any_success = df.loc[df["success"] == True, "task_id"].nunique()
    total_tasks = df["task_id"].nunique()
    print(f"\n   Distinct tasks solved at ANY budget: {any_success}/{total_tasks}"
          f" ({any_success/total_tasks:.0%})")
    
    return tbl


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


def task_solvability(df: pd.DataFrame) -> pd.DataFrame:
    """Which tasks are solvable, and at which minimum budget?"""
    print_section("Task Solvability")
    
    solved = df[df["success"] == True].groupby("task_id").agg(
        site=("site", "first"),
        min_budget=("budget_level", "min"),
        max_budget=("budget_level", "max"),
        times_solved=("success", "sum"),
        budgets_tested=("budget_level", "nunique"),
    ).sort_values("min_budget")
    
    total_tasks = df["task_id"].nunique()
    print(f"   {len(solved)}/{total_tasks} tasks solved at least once "
          f"({len(solved)/total_tasks:.0%})\n")
    
    if len(solved):
        print(solved.to_string())
        print(f"\n   Minimum budget needed (median of solved tasks): "
              f"{solved['min_budget'].median():.0f}")
    else:
        print("   No tasks solved at any budget level.")
    
    return solved


def termination_reasons(df: pd.DataFrame) -> pd.Series:
    print_section("Termination Reasons")
    counts = df["termination_reason"].value_counts()
    total = len(df)
    for reason, n in counts.items():
        print(f"   {reason:30s} {n:4d}  ({n/total:.0%})")
    return counts


def token_distribution(df: pd.DataFrame) -> None:
    print_section("Token Distribution")
    toks = df["total_tokens"].dropna()
    print(f"episodes           : {len(toks)}")
    print(f"min                : {toks.min():.0f}")
    print(f"25th percentile    : {toks.quantile(0.25):.0f}")
    print(f"median             : {toks.median():.0f}")
    print(f"75th percentile    : {toks.quantile(0.75):.0f}")
    print(f"max                : {toks.max():.0f}")
    succ = df.loc[df["success"] == True, "total_tokens"]
    if len(succ):
        print(f"   median of successes: {succ.median():.0f}  (n={len(succ)})")


def per_step_cost(df: pd.DataFrame) -> None:
    print_section("Per-step Token Cost")
    df_valid = df[df["steps"] > 0].copy()
    df_valid["tok_per_step"] = df_valid["total_tokens"] / df_valid["steps"]
    by_site = df_valid.groupby("site")["tok_per_step"].agg(["median", "mean", "std"])
    print(by_site.to_string())
    overall = df_valid["tok_per_step"]
    print(f"\n   overall median: {overall.median():.0f}  "
          f"mean: {overall.mean():.0f}  std: {overall.std():.0f}")


def difficulty_breakdown(df: pd.DataFrame) -> None:
    """Only runs if tier data has been joined."""
    if "difficulty_tier" not in df.columns:
        return
    print_section("Success Rate by Difficulty Tier (RQ2)")
    tier_order = ["Easy", "Medium", "Hard"]
    df["difficulty_tier"] = pd.Categorical(
        df["difficulty_tier"], categories=tier_order, ordered=True
    )
    
    rows = []
    for (tier, budget), grp in df.groupby(["difficulty_tier", "budget_level"]):
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