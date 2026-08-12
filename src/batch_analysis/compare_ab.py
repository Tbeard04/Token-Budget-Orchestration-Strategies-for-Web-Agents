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


def cost_ratio(a: pd.DataFrame, b: pd.DataFrame) -> None:
    print_section("A vs B: Cost Ratio on Shared Successes")
 
    a_solved = set(a.loc[a["success"] == True, "task_id"])
    b_solved = set(b.loc[b["success"] == True, "task_id"])
    both = a_solved & b_solved
 
    if not both:
        print("No shared successes to compare.")
        return
 
    ratios = []
    for tid in both:
        a_cost = a.loc[(a["task_id"] == tid) & (a["success"] == True), "total_tokens"].min()
        b_cost = b.loc[(b["task_id"] == tid) & (b["success"] == True), "total_tokens"].min()
        if a_cost and a_cost > 0:
            ratios.append(b_cost / a_cost)
 
    if not ratios:
        return
 
    print(f"tasks solved by both: {len(ratios)}")
    print(f"median B/A cost ratio: {statistics.median(ratios):.2f}x")
    print(f"mean B/A cost ratio: {statistics.mean(ratios):.2f}x")
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
 
    print(f"solved by both   : {len(both)}")
    print(f"solved by A only : {len(a_only)}")
    print(f"solved by B only : {len(b_only)}")
    print(f"solved by neither: {len(neither)}")
 
    if b_only:
        print(f"\n   Tasks B solved that A could not "
              f"(the pipeline's capability advantage):")
        with_rev = 0
        for tid in sorted(b_only):
            eps = b[(b["task_id"] == tid) & (b["success"] == True)]
            if eps.empty:
                continue
            ep = eps.loc[eps["budget_level"].idxmin()]
            revs = int(ep.get("critic_revisions", 0))
            if revs:
                with_rev += 1
            # What did A do on this task at its best budget?
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
    """Does the pipeline change HOW episodes fail, not just how often?"""
    print_section("A vs B: Failure Mode Shift")
 
    def profile(df, label):
        fails = df[df["success"] == False]
        if fails.empty:
            return
        budget = fails["termination_reason"].isin(BUDGET_TERMINATIONS).sum()
        stuck = fails["termination_reason"].isin(STUCK_TERMINATIONS).sum()
        wrong = (fails["termination_reason"] == "env_terminated").sum()
        n = len(fails)
        print(f"   {label}: {n} failures")
        print(f"     budget exhausted : {budget:>5}  ({budget/n:.0%})")
        print(f"     stuck (guards)   : {stuck:>5}  ({stuck/n:.0%})")
        print(f"     wrong answer     : {wrong:>5}  ({wrong/n:.0%})")
 
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
 

