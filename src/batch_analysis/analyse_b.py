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