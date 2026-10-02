"""
select_tasks.py - pick the fixed task set
Read-only tasks only: every task is attempted 9 times (3 strategies x 3 spending schemes)
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from spending_schemes import spending_scheme_config as C

#paths to the metadata and risk files
METADATA = Path("../data/processed/final_annotation_difficulty_tiers/task_metadata.jsonl")
#path to the risk file
RISK = Path("../data/processed/task_list/task_risk_levels.jsonl")
#path to the output file
OUT = Path("../data/processed/task_list/read_only_tasks.jsonl")

#main function
def main() -> None:
    #load the metadata
    meta = {r["task_id"]: r for r in map(json.loads, METADATA.read_text().splitlines()) if r}
    #load the risk
    risk = {r["task_id"]: r for r in map(json.loads, RISK.read_text().splitlines()) if r}

    #create a dictionary to store the tasks by difficulty tier
    pool: dict[str, list[int]] = defaultdict(list)
    #create a dictionary to store the excluded tasks
    excluded = {"not_read_only": 0, "impossible": 0}
    #for each task in the risk
    for tid, r in risk.items():
        if r["risk_level"] != "read_only":
            excluded["not_read_only"] += 1
            continue
        if "impossible" in r.get("risk_reason", ""):
            excluded["impossible"] += 1
            continue
        pool[meta[tid]["difficulty_tier"]].append(tid)

    rng = random.Random(C.TASK_SEED)
    chosen = []
    for tier, n in C.TIER_SPLIT.items():
        ids = sorted(pool[tier])
        if len(ids) < n:
            raise SystemExit(f"only {len(ids)} eligible {tier} tasks, need {n}")
        chosen += [{"task_id": t, "site": meta[t]["site"], "difficulty_tier": tier}
                   for t in sorted(rng.sample(ids, n))]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for t in chosen:
            fh.write(json.dumps(t) + "\n")

    print(f"[select] eligible read-only, non-impossible: "
          + ", ".join(f"{t} {len(v)}" for t, v in sorted(pool.items())))
    print(f"[select] excluded: {excluded}")
    print(f"[select] chose {len(chosen)} tasks --> {OUT}")

if __name__ == "__main__":
    main()