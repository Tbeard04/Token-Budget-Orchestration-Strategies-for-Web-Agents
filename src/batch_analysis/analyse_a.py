from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import pandas as pd

from batch_analysis.shared import (
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
        print("no episodes with steps > 0")
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


def answer_failure_analysis(df: pd.DataFrame) -> None:
    print_section("Answer Failures (potential Critic value)")

    wrong = df[(df["success"] == False) &
               (df["termination_reason"] == "env_terminated")]
    if wrong.empty:
        print("No wrong-answer terminations.")
        return

    print(f" Episodes ending in a wrong answer: {len(wrong)} "
          f"({len(wrong)/len(df):.0%} of all episodes)")
    print(f" Median tokens spent before answering: {wrong['total_tokens'].median():.0f}")
    print(f" Median steps before answering: {wrong['steps'].median():.0f}")

    print("\n By budget level:")
    for budget, grp in wrong.groupby("budget_level"):
        pct = len(grp) / len(df[df["budget_level"] == budget])
        print(f"{budget:>6}: {len(grp):>4} episodes ({pct:.0%})")

    print("\n By site:")
    for site, grp in wrong.groupby("site"):
        print(f" {site:16s}: {len(grp):>4} episodes")


def budget_utilisation(df: pd.DataFrame) -> None:
    print_section("Budget Utilisation")

    df = df.copy()
    df["utilisation"] = df["total_tokens"] / df["budget_level"]

    print("budget  |  all eps  |  successes  |  failures")
    for budget, grp in df.groupby("budget_level"):
        succ = grp[grp["success"] == True]["utilisation"]
        fail = grp[grp["success"] == False]["utilisation"]
        succ_str = f"{succ.median():.0%}" if len(succ) else "  - "
        fail_str = f"{fail.median():.0%}" if len(fail) else "  - "
        print(f" {budget:>6}  |   {grp['utilisation'].median():>5.0%}   |"
              f" {succ_str:>5}    |   {fail_str:>5}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="Strategy A JSONL")
    ap.add_argument("--tiers", default=None, help="task_metadata.jsonl")
    ap.add_argument("--out", default="../data/processed/strategy_a")
    ap.add_argument("--verbose-tasks", action="store_true", help="print the full per-task solvability table")
    args = ap.parse_args()

    df = load(args.file)
    print(f"Loaded {len(df)} Strategy A episodes from {args.file}")
    df = join_tiers(df, args.tiers)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    #Shared analysis
    run_shared_analysis(df, out_dir, verbose_tasks=args.verbose_tasks)

    #Strategy A specific
    budget_as_binding_constraint(df)
    cost_floor_profile(df)
    answer_failure_analysis(df)
    budget_utilisation(df)

    print(f"\nDone. {len(df)} episodes analysed.")


if __name__ == "__main__":
    main()
