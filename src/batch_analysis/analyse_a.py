from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import pandas as pd

from analysis.shared import (
    load, join_tiers, print_section, run_shared_analysis,
    BUDGET_TERMINATIONS, STUCK_TERMINATIONS,
)


def budget_as_binding_constraint(df: pd.DataFrame) -> None:
    print_section("Budget as Binding Constraint")

    failures = df[df["success"] == False]
    if failures.empty:
        print("No failures.")
        return

    budget_driven = failures["termination_reason"].isin(BUDGET_TERMINATIONS).sum()
    stuck = failures["termination_reason"].isin(STUCK_TERMINATIONS).sum()
    other = len(failures) - budget_driven - stuck

    print(f"total failures: {len(failures)}")
    print(f"budget-driven: {budget_driven:5d}  ({budget_driven/len(failures):.0%})")
    print(f"stuck (guards): {stuck:5d}  ({stuck/len(failures):.0%})")
    print(f"other (wrong answer, env terminated): {other:5d}  ({other/len(failures):.0%})")

    print("\n By budget level:")
    for budget, grp in failures.groupby("budget_level"):
        bd = grp["termination_reason"].isin(BUDGET_TERMINATIONS).sum()
        print(f" {budget:>6}: {bd:>4}/{len(grp):<4} budget-driven ({bd/len(grp):.0%})")


def cost_floor_profile(df: pd.DataFrame) -> None:
    print_section("Cost Floor Profile (Strategy C target)")

    valid = df[df["steps"] > 0].copy()
    if valid.empty:
        print("   no episodes with steps > 0")
        return

    valid["tok_per_step"] = valid["total_tokens"] / valid["steps"]

    print(f" median tokens per step (all): {valid['tok_per_step'].median():.0f}")
    succ = valid[valid["success"] == True]
    if len(succ):
        print(f" median tokens per step (successes): {succ['tok_per_step'].median():.0f}")
        print(f" median total tokens to succeed : {succ['total_tokens'].median():.0f}")
        print(f" median steps to succeed: {succ['steps'].median():.0f}")

    print("\n Per-step cost by budget level:")
    for budget, grp in valid.groupby("budget_level"):
        print(f" {budget:>6}: {grp['tok_per_step'].median():>6.0f} tokens/step")
    # Strategy C should match this per-step cost when routing to the Executor alone, and exceed it only when invoking more agents.

