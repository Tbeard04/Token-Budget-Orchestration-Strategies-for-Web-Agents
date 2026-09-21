"""
flag_episodes.py - decide which episodes Strategy C may learn a reward from.
"""
from __future__ import annotations
 
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
 
RISK_LEVELS = ["read_only", "idempotent", "non_idempotent"]
 
 
def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows

def demanding(meta: dict | None) -> bool:
    #if the metadata is not provided
    if not meta:
        #no metadata: fall back to flagging everything
        return True
    #if the interaction is 2 or the difficulty tier is Medium or Hard
    return (meta.get("interaction") == 2
            or meta.get("difficulty_tier") in ("Medium", "Hard"))


def flag(episodes: list[dict], risk: dict, lenient_idempotent: bool = False, pre_steps: int = 1, metadata: dict | None = None) -> list[dict]:

    #if the metadata is not provided, use an empty dictionary
    metadata = metadata or {}
    #iterate over the episodes and remember the input order
    for i, e in enumerate(episodes):
        e["_i"] = i
 
    #create a dictionary to store the episodes by strategy and task_id
    groups: dict[tuple, list[dict]] = defaultdict(list)
    #iterate over the episodes and add them to the dictionary
    for e in episodes:
        groups[(e.get("strategy"), e.get("task_id"))].append(e)
 
    #iterate over the episodes and add them to the dictionary
    for (strategy, task_id), eps in groups.items():
        #get the level
        level = risk.get(task_id, "unknown")
        #sort the episodes by timestamp and input order
        eps.sort(key=lambda e: (e.get("timestamp", ""), e["_i"]))
        #get the first success
 
        first = next((i for i, e in enumerate(eps) if e.get("success")), None)
        fs_budget = eps[first].get("budget_level") if first is not None else None
        fs_steps = eps[first].get("steps") if first is not None else None
 
        #if the level is idempotent or non_idempotent and the first success is not None and the steps are less than or equal to the pre_steps and the metadata is demanding
        suspect = (level in ("idempotent", "non_idempotent")
                   and first is not None
                   and (fs_steps or 0) <= pre_steps
                   and demanding(metadata.get(task_id)))
 
        exempt = level == "read_only" or (lenient_idempotent and level == "idempotent")
        #iterate over the episodes and add the metadata
        for idx, e in enumerate(eps):
            after = first is not None and idx > first
            excluded = after and not exempt

            #add the metadata to the episode
            e["risk_level"] = level
            e["episode_index"] = idx
            e["first_success_index"] = first
            e["first_success_budget"] = fs_budget
            e["suspect_pre_existing"] = suspect
            e["use_for_routing"] = True
            e["use_for_reward"] = not excluded
            if not excluded:
                e["contamination_reason"] = None
            elif level == "idempotent":
                e["contamination_reason"] = "idempotent_post_first_success"
            else:
                e["contamination_reason"] = "post_first_success"
 
    for e in episodes:
        del e["_i"]
    return episodes