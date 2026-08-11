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
