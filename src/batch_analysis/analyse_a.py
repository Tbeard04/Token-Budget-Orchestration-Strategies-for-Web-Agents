from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import pandas as pd

from batch_analysis.shared import (load, join_tiers, print_section, run_shared_analysis, BUDGET_TERMINATIONS, STUCK_TERMINATIONS,)

#Budget as Binding Constraint
def budget_as_binding_constraint(df: pd.DataFrame) -> None:
    print_section("Budget as Binding Constraint")
    #Failures are episodes that did not succeed
    failures = df[df["success"] == False]
    if failures.empty:
        print("No failures.")
        return

    #Budget-driven failures are episodes that terminated because the budget was exceeded
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

#Cost Floor Profile (Strategy C target)
def cost_floor_profile(df: pd.DataFrame) -> None:
    print_section("Cost Floor Profile (Strategy C target)")
    #Valid episodes are episodes that had steps > 0
    valid = df[df["steps"] > 0].copy()
    if valid.empty:
        print("no episodes with steps > 0")
        return

    #Tokens per step is the total tokens divided by the number of steps
    valid["tok_per_step"] = valid["total_tokens"] / valid["steps"]
    #Print the median tokens per step for all episodes
    print(f" median tokens per step (all): {valid['tok_per_step'].median():.0f}")
    #Successes are episodes that succeeded
    succ = valid[valid["success"] == True]
    #Print the median tokens per step for successes
    if len(succ):
        print(f"median tokens per step (successes): {succ['tok_per_step'].median():.0f}")
        #Print the median total tokens to succeed
        print(f"median total tokens to succeed : {succ['total_tokens'].median():.0f}")
        #Print the median steps to succeed
        print(f"median steps to succeed: {succ['steps'].median():.0f}")

    print("\n Per-step cost by budget level:")
    for budget, grp in valid.groupby("budget_level"):
        print(f" {budget:>6}: {grp['tok_per_step'].median():>6.0f} tokens/step")
    # Strategy C should match this per-step cost when routing to the Executor alone, and exceed it only when invoking more agents


def answer_failure_analysis(df: pd.DataFrame) -> None:
    print_section("Answer Failures (potential Critic value)")
    #Wrong are episodes that did not succeed and terminated because of a wrong answer
    wrong = df[(df["success"] == False) &
               (df["termination_reason"] == "env_terminated")]
    if wrong.empty:
        print("No wrong-answer terminations.")
        return

    print(f"Episodes ending in a wrong answer: {len(wrong)} "
          f"({len(wrong)/len(df):.0%} of all episodes)")
    print(f"Median tokens spent before answering: {wrong['total_tokens'].median():.0f}")
    print(f"Median steps before answering: {wrong['steps'].median():.0f}")

    print("\n By budget level:")
    for budget, grp in wrong.groupby("budget_level"):
        pct = len(grp) / len(df[df["budget_level"] == budget])
        print(f"{budget:>6}: {len(grp):>4} episodes ({pct:.0%})")

    print("\n By site:")
    for site, grp in wrong.groupby("site"):
        print(f" {site:16s}: {len(grp):>4} episodes")

#Budget Utilisation
def budget_utilisation(df: pd.DataFrame) -> None:
    print_section("Budget Utilisation")
    #Copy the dataframe
    df = df.copy()
    df["utilisation"] = df["total_tokens"] / df["budget_level"]
    #Print the budget, all episodes, successes, and failures
    print("budget  |  all eps  |  successes  |  failures")
    for budget, grp in df.groupby("budget_level"):
        succ = grp[grp["success"] == True]["utilisation"]
        #Successes are episodes that succeeded
        fail = grp[grp["success"] == False]["utilisation"]
        #Failures are episodes that did not succeed
        succ_str = f"{succ.median():.0%}" if len(succ) else "  - "
        #Print the median utilisation for successes
        fail_str = f"{fail.median():.0%}" if len(fail) else "  - "
        #Print the median utilisation for failures
        print(f" {budget:>6} | {grp['utilisation'].median():>5.0%} |"
              f" {succ_str:>5} | {fail_str:>5}")

#Main function
def main() -> None:
    #Create an argument parser
    ap = argparse.ArgumentParser()
    #Add an argument for the file
    ap.add_argument("--file", required=True, help="Strategy A JSONL")
    #Add an argument for the tiers
    ap.add_argument("--tiers", default=None, help="task_metadata.jsonl")
    #Add an argument for the output directory
    ap.add_argument("--out", default="../data/processed/strategy_a")
    ap.add_argument("--verbose-tasks", action="store_true", help="print the full per-task solvability table")
    args = ap.parse_args()

    #Load the dataframe
    df = load(args.file)
    #Print the number of episodes loaded
    print(f"Loaded {len(df)} Strategy A episodes from {args.file}")
    #Join the tiers
    df = join_tiers(df, args.tiers)
    #create the output directory
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