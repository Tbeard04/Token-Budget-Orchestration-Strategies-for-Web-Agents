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

