"""
This script is used to join the episodes with the metadata and risk levels.
"""
# Join both A, B and metadata to create single training dataset for C. 
# 4. Join the datasets on the task_id column
# 5. Save the joined dataset to a new file in the processed directory
# 6. each row needs to contain both A and B - IMPORTANT 
# split function to allow for different splits of the dataset (for analysis comparing difficulty tiers etc.. between A & B tasks) --> probably a seperate script for this


from __future__ import annotations
 
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
 
#Removed from every episode. Everything else is passed through
DROP_FIELDS = {"step_log"}
 
#attached from task_metadata.jsonl
#task_id is the join key
META_FIELDS = [
    "task_category",
    "pages_to_traverse",
    "retrieval_type",
    "interaction",
    "target_locatability",
    "rubric_total",
    "difficulty_tier",
    "label_source",
]
 
#Attached from task_risk_levels.jsonl
#risk_level is already on the episode
RISK_FIELDS = ["risk_reason"]
 
#Task-level fields that also exist on the episode
CROSS_CHECK = {"site": "site", "goal": "intent"}
 
#Budgets for the tasks
BUDGETS = [2000, 4000, 8000, 16000, 32000, 64000]
 
 
def load_jsonl(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


#the join function
def join(episodes: list[dict],
         meta: dict[int, dict],
         risk: dict[int, dict],
         keep_steps: bool = False) -> tuple[list[dict], dict]:
    #if keep_steps is True, the steps will be kept, otherwise they will be dropped
    drop = set() if keep_steps else DROP_FIELDS
    #report is a dictionary that will be used to store the missing metadata and risk, as well as the conflicts
    report = {
        "missing_meta": set(),
        "missing_risk": set(),
        "conflicts": defaultdict(list),
        "dropped_fields": sorted(drop),
    }
 
    out = [] #output list to store the joined episodes
    for e in episodes:
        tid = e["task_id"]
        m = meta.get(tid)
        r = risk.get(tid)
        if m is None:
            report["missing_meta"].add(tid)
        if r is None:
            report["missing_risk"].add(tid)
 
        row = {k: v for k, v in e.items() if k not in drop}
 
        #integrity: the episode and the metadata must agree about the task
        if m is not None:
            for ep_key, meta_key in CROSS_CHECK.items():
                a, b = e.get(ep_key), m.get(meta_key)
                if a is not None and b is not None and a != b:
                    report["conflicts"][ep_key].append(tid)
 
        if m is not None:
            for f in META_FIELDS:
                row[f] = m.get(f)
        else:
            for f in META_FIELDS:
                row[f] = None
 
        if r is not None:
            for f in RISK_FIELDS:
                row[f] = r.get(f)
        else:
            for f in RISK_FIELDS:
                row[f] = None
 
        out.append(row)
    #sort the missing metadata and risk, and the conflicts
    report["missing_meta"] = sorted(report["missing_meta"])
    report["missing_risk"] = sorted(report["missing_risk"])
    report["conflicts"] = {k: sorted(set(v)) for k, v in report["conflicts"].items()}
    return out, report