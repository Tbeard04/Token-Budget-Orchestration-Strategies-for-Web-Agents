#test
## add spending schemes analysis here for shared analysis between A,B & C
## 1. pay-as-you-go which is the full batches completed already 
## IMPORTANT NOTE: READ-ONLY TASKS (100 tasks) SO THEY ARE NOT CONTAMINATED BY MODIFIED STATE TASKS
## 2. front loaded = the first step can use up to 40% of the budget, and the later steps share the rest
## 3. Even scheme = the budget is split equally over the number of steps that strategy usually takes at 64k
## 4. Reactive scheme = each step starts with the even share, gets 1.5× after a failed move or when the page didn't change, and 0.75× after a move that made progress
#test

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from batch_analysis.shared import (load, print_section, wilson, mcnemar_exact, outcome_group, COLOURS, MARKERS, NAMES, TIER_ORDER, OUTCOME_ORDER, OUTCOME_COLOURS)

BUDGET = 32_000
STRATEGIES = ["A", "B", "C"]
SCHEMES = ["pay_as_you_go", "even", "front_loaded", "reactive"]
SCHEME_LABELS = {"pay_as_you_go": "Pay as you go", "even": "Even", "front_loaded": "Front-loaded", "reactive": "Reactive"}

#Loading
#scheme runs + the pay-as-you-go reference from the main batches, on the same tasks
def load_grid(args) -> pd.DataFrame:
    sch = load(args.schemes)
    sch = sch[sch["budget_level"] == BUDGET]
    tasks = set(sch["task_id"])
    parts = [sch]
    for s, path in [("A", args.main_a), ("B", args.main_b), ("C", args.main_c)]:
        if not path:
            continue
        m = load(path)
        m = m[(m["budget_level"] == BUDGET) & (m["task_id"].isin(tasks))].copy()
        m["strategy"] = s
        m["scheme"] = "pay_as_you_go"
        parts.append(m)
    df = pd.concat(parts, ignore_index=True, sort=False)

    #difficulty tier per task: from the task list if given, else from the main-batch files
    tiers = None
    if args.tasks and Path(args.tasks).exists():
        rows = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
        tiers = {r["task_id"]: r.get("difficulty_tier") for r in rows}
    elif "difficulty_tier" in df.columns:
        tiers = df.dropna(subset=["difficulty_tier"]).drop_duplicates("task_id").set_index("task_id")["difficulty_tier"].to_dict()
    if tiers:
        df["difficulty_tier"] = df["task_id"].map(tiers)

    #scheme-run mechanics, pulled out of step_log (absent for pay-as-you-go rows)
    def step_stats(steps):
        #main-batch step logs have no scheme fields: not applicable rather than 0
        if not isinstance(steps, list) or not steps or "trimmed" not in steps[0]:
            return pd.Series({"trim_share": np.nan, "high_share": np.nan})
        by_step = {}
        #get the step stats
        for s in steps:
            by_step.setdefault(s["step"], s)
        return pd.Series({"trim_share": np.mean([bool(s.get("trimmed")) for s in by_step.values()]), "high_share": np.mean([s.get("effort") == "high" for s in by_step.values()])})
    if "step_log" in df.columns:
        df = pd.concat([df, df["step_log"].apply(step_stats)], axis=1)
 
    #scheme categories
    df["scheme"] = pd.Categorical(df["scheme"], categories=[s for s in SCHEMES if s in set(df["scheme"])], ordered=True)
    df["outcome"] = df["termination_reason"].map(outcome_group)
    return df

