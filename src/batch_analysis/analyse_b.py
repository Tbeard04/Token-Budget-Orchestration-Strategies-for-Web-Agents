from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from analysis.shared import (
    load, join_tiers, print_section, run_shared_analysis, COLOURS,
)

# Extraction helpers

def _critic_steps(df: pd.DataFrame):
    for _, row in df.iterrows():
        steps = row.get("step_log")
        if not isinstance(steps, list):
            continue
        for s in steps:
            if s.get("agent_role") == "critic":
                yield row, s


def _extract_revisions(df: pd.DataFrame) -> list[dict]:
    revisions = []
    for row, s in _critic_steps(df):
        if not s.get("revised"):
            continue
        steps = row["step_log"]
        step_n = s["step"]

        #State at the time of the revision
        prev = [x for x in steps if x["step"] == step_n - 1]
        prev_error = any(x.get("action_error") not in (None, "None") for x in prev) if prev else False

        #Did the revised action work
        nxt = [x for x in steps if x["step"] == step_n + 1]
        if nxt:
            next_error = any(x.get("action_error") not in (None, "None") for x in nxt)
            revision_worked = not next_error
        else:
            revision_worked = False # episode ended after this step

        revisions.append({
            "task_id": row.get("task_id"),
            "budget_level": row.get("budget_level"),
            "site": row.get("site"),
            "step": step_n,
            "budget_frac_remaining": 1.0 - (s.get("cumulative_tokens", 0) / max(row.get("budget_level", 1), 1)),
            "prev_step_had_error": prev_error,
            "proposed": s.get("proposed_action", "?"),
            "revised_to": s.get("action", "?"),
            "reasoning": s.get("critic_reasoning", "?"),
            "revision_worked": revision_worked,
            "episode_success": bool(row.get("success", False)),
            "difficulty_tier": row.get("difficulty_tier", "Unknown"),
        })
    return revisions


# ── Critic tables ───────────────────────────────────────────────────────

def critic_analysis(df: pd.DataFrame) -> None:
    print_section("Critic Analysis: Rate and Outcome")

    total_calls = sum(1 for _ in _critic_steps(df))
    total_revisions = int(df["critic_revisions"].sum())
    eps_with = int((df["critic_revisions"] > 0).sum())

    print(f" episodes: {len(df)}")
    print(f" episodes with revisions: {eps_with}  ({eps_with/len(df):.1%})")
    print(f" total critic calls: {total_calls}")
    print(f" total revisions: {total_revisions}")
    if total_calls:
        print(f" revision rate: {total_revisions/total_calls:.1%}")
        print(f" rubber-stamp rate: "
              f"{(total_calls-total_revisions)/total_calls:.1%}")

    revised = df[df["critic_revisions"] > 0]
    unrevised = df[df["critic_revisions"] == 0]
    r_sr = revised["success"].mean() if len(revised) else 0
    u_sr = unrevised["success"].mean() if len(unrevised) else 0

    print(f"\n SR of episodes WITH revisions   : {r_sr:.1%}  (n={len(revised)})")
    print(f" SR of episodes WITHOUT revisions: {u_sr:.1%}  (n={len(unrevised)})")

    print("\n By budget level:")
    for budget, grp in df.groupby("budget_level"):
        calls = sum(1 for _ in _critic_steps(grp))
        revs = int(grp["critic_revisions"].sum())
        rate = revs / calls if calls else 0
        print(f" {budget:>6}: {calls:>5} calls  {revs:>3} revisions  ({rate:.1%})")


def revision_state_profile(df: pd.DataFrame) -> None:
    print_section("Revision State Profile (Strategy C routing signal)")

    revisions = _extract_revisions(df)
    if not revisions:
        print("No revisions to profile.")
        return

    rev = pd.DataFrame(revisions)
    n = len(rev)

    print(f"revisions analysed: {n}\n")

    with_err = rev["prev_step_had_error"].sum()
    print(f" preceded by an action error : {with_err:>4}  ({with_err/n:.0%})")
    print(f" preceded by a clean step: {n-with_err:>4}  ({(n-with_err)/n:.0%})")

    all_calls = 0
    calls_after_error = 0
    for row, s in _critic_steps(df):
        all_calls += 1
        steps = row["step_log"]
        prev = [x for x in steps if x["step"] == s["step"] - 1]
        if prev and any(x.get("action_error") not in (None, "None") for x in prev):
            calls_after_error += 1
    base_rate = calls_after_error / all_calls if all_calls else 0
    print(f"\n base rate of post-error critic calls: {base_rate:.0%}")
    if base_rate:
        lift = (with_err / n) / base_rate
        print(f" lift from conditioning on action_error: {lift:.2f}x")
        if lift > 1.3:
            print(" action_error is a STRONG predictor of a useful Critic call.")
            print(" Strategy C should invoke the Critic after errors.")
        else:
            print(" action_error is a weak predictor. Revisions happen in")
            print(" clean states too, which makes them harder to route for.")

    print(f"\n Step number at revision:")
    print(f" median: {rev['step'].median():.0f}   "
          f"min: {rev['step'].min()}   max: {rev['step'].max()}")

    print(f"\n Budget remaining at revision:")
    print(f" median: {rev['budget_frac_remaining'].median():.0%}")

    worked = rev["revision_worked"].sum()
    print(f"\n Revised action executed without error: {worked}/{n} ({worked/n:.0%})")
    led_to_success = rev["episode_success"].sum()
    print(f" Revisions in episodes that succeeded : {led_to_success}/{n} "
          f"({led_to_success/n:.0%})")

    print("\n By site:")
    for site, grp in rev.groupby("site"):
        w = grp["revision_worked"].sum()
        print(f" {site:16s}: {len(grp):>3} revisions, {w} worked, "
              f"{int(grp['episode_success'].sum())} in successful episodes")

    if "difficulty_tier" in rev.columns and rev["difficulty_tier"].notna().any():
        print("\n By difficulty tier:")
        for tier, grp in rev.groupby("difficulty_tier"):
            print(f" {str(tier):10s}: {len(grp):>3} revisions")


def critic_revision_details(df: pd.DataFrame, limit: int | None = 20) -> None:
    revisions = _extract_revisions(df)
    if not revisions:
        return
    print_section(f"Critic Revision Details"
                  f"{f' (first {limit})' if limit else ''}")

    shown = revisions if limit is None else revisions[:limit]
    for i, r in enumerate(shown, 1):
        outcome = "SUCCESS" if r["episode_success"] else "failed"
        worked = "worked" if r["revision_worked"] else "also failed"
        err = "after error" if r["prev_step_had_error"] else "clean state"
        print(f"\n   [{i}] task {r['task_id']} @ {r['budget_level']} "
              f"step {r['step']}  ({r['site']}, {err})")
        print(f" proposed: {str(r['proposed'])[:66]}")
        print(f" revised: {str(r['revised_to'])[:66]}")
        print(f" outcome: revision {worked}, episode {outcome}")
        print(f" reason: {str(r['reasoning'])[:110]}")

def rubber_stamp_cost(df: pd.DataFrame) -> None:
    print_section("Rubber-Stamp Cost (Strategy C savings ceiling)")

    if "tokens_by_role" not in df.columns:
        print("   tokens_by_role not present.")
        return

    total_calls = 0
    approval_calls = 0
    critic_tokens_total = 0
    approval_tokens = 0

    for row, s in _critic_steps(df):
        total_calls += 1
        call_tokens = s.get("input_tokens", 0) + s.get("output_tokens", 0)
        critic_tokens_total += call_tokens
        if not s.get("revised"):
            approval_calls += 1
            approval_tokens += call_tokens

    episode_tokens = df["total_tokens"].sum()

    print(f" total critic calls        : {total_calls}")
    print(f" approval-only calls       : {approval_calls} "
          f"({approval_calls/max(total_calls,1):.0%})")
    print(f"\n all critic tokens         : {critic_tokens_total:,}")
    print(f" tokens on approvals only  : {approval_tokens:,}")
    print(f" total episode tokens      : {episode_tokens:,}")
    print(f"\n critic share of all spend : "
          f"{critic_tokens_total/max(episode_tokens,1):.1%}")
    print(f" RECOVERABLE by perfect routing: "
          f"{approval_tokens/max(episode_tokens,1):.1%} of all tokens")

