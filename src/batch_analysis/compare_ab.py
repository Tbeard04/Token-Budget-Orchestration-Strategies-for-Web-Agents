"""
analysis/compare_ab.py - Strategy A vs Strategy B comparison.

Produces the figures for RQ1 and the numbers that motivate
Strategy C:

  - comparison_table: SR side by side at each budget
  - equivalent_budget: what budget B needs to match A
  - cost_ratio: how much more B costs on shared successes
  - task_level_comparison: which tasks each solved that the other did not
  - strategy_c_ceiling: what a perfect router could achieve
  - failure_mode_shift: does the pipeline change HOW episodes fail

Run from src/:
    python -m analysis.compare_ab \ --a ../data/raw/strategy_a.jsonl \ --b ../data/raw/strategy_b.jsonl
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis.shared import (
    load, join_tiers, print_section, COLOURS,
    BUDGET_TERMINATIONS, STUCK_TERMINATIONS,
)

def comparison_table(a: pd.DataFrame, b: pd.DataFrame) -> pd.DataFrame:
    print_section("A vs B: Success Rate by Budget")
 
    rows = []
    for budget in sorted(set(a["budget_level"]) | set(b["budget_level"])):
        ag = a[a["budget_level"] == budget]
        bg = b[b["budget_level"] == budget]
        rows.append({
            "budget": budget,
            "A_n": len(ag),
            "A_SR": ag["success"].mean() if len(ag) else float("nan"),
            "A_med_tok": ag["total_tokens"].median() if len(ag) else float("nan"),
            "B_n": len(bg),
            "B_SR": bg["success"].mean() if len(bg) else float("nan"),
            "B_med_tok": bg["total_tokens"].median() if len(bg) else float("nan"),
        })
    tbl = pd.DataFrame(rows).set_index("budget")
    tbl["SR_gap"] = tbl["B_SR"] - tbl["A_SR"]
 
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(tbl.to_string())
    return tbl


def equivalent_budget(a: pd.DataFrame, b: pd.DataFrame) -> None:
    print_section("A vs B: Equivalent Budget")
 
    a_sr = a.groupby("budget_level")["success"].mean()
    b_sr = b.groupby("budget_level")["success"].mean()
 
    for b_budget, b_val in b_sr.items():
        closest = (a_sr - b_val).abs().idxmin()
        ratio = b_budget / closest if closest else float("inf")
        print(f"   B @ {b_budget//1000:>2}k ({b_val:>5.1%})  "
              f"~=  A @ {closest//1000:>2}k ({a_sr[closest]:>5.1%})   "
              f"[B needs {ratio:.1f}x the budget]")

