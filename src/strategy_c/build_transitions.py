"""
build_transitions.py - turn final_episodes.jsonl into the two jsonl files the Strategy C router trains on
"""

from __future__ import annotations
 
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
 
ACTION_OF = {"A": "execute", "B": "cycle"}
TIER_ORD = {"Easy": 0, "Medium": 1, "Hard": 2}
 
TASK_FIELDS = ["site", "task_category", "risk_level", "difficulty_tier", "pages_to_traverse", "retrieval_type", "interaction", "target_locatability", "rubric_total"]


#load the jsonl file
def load_jsonl(path: str | Path) -> list[dict]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out
 
 #check if the action_error is not None, "None", or empty
def _has_error(entry: dict) -> bool:
    return entry.get("action_error") not in (None, "None", "")
 
 
def _by_step(step_log: list[dict]) -> list[list[dict]]:
    #group the log entries by website step, in step order
    groups: dict[int, list[dict]] = defaultdict(list)
    for e in step_log:
        groups[e["step"]].append(e)
    return [groups[k] for k in sorted(groups)]