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



#transitions
#one row per site step + a terminal stop row
def episode_rows(ep: dict) -> list[dict]:
    budget = ep["budget_level"]
    mode = ep.get("mode_chosen") or ACTION_OF.get(ep["strategy"], ep["strategy"])
    base = {
        "task_id": ep["task_id"],
        "strategy": ep["strategy"],
        "mode": mode,
        "budget_level": budget,
        "episode_success": bool(ep["success"]),
        "episode_tokens": ep["total_tokens"],
        "use_for_reward": ep.get("use_for_reward", True),
        **{f: ep.get(f) for f in TASK_FIELDS},
    }
    base["tier_ord"] = TIER_ORD.get(ep.get("difficulty_tier"))
 
    rows = []
    cum_before = 0
    last_error = False
    url_changed = False
    consecutive_errors = 0
 
    #loop through the steps
    for k, calls in enumerate(_by_step(ep.get("step_log") or [])):
        #get the maximum cumulative tokens
        cum_after = max(c.get("cumulative_tokens", cum_before) for c in calls)
        #add the row to the list
        rows.append({
            **base,
            "step_index": k,
            "action": mode,
            "budget_remaining_frac": round(1 - cum_before / budget, 4),
            "last_error": int(last_error),
            "url_changed_last": int(url_changed),
            "consecutive_errors": consecutive_errors,
            "step_cost": cum_after - cum_before,
            "is_terminal": False,
            "termination_reason": None,
        })
        #check if there was an error
        last_error = any(_has_error(c) for c in calls)
        #check if the url changed
        url_changed = any(c.get("url") != c.get("next_url") for c in calls)
        #check if there were consecutive errors
        consecutive_errors = consecutive_errors + 1 if last_error else 0
        #update the cumulative tokens
        cum_before = cum_after
 
    #add the terminal stop row
    rows.append({
        **base,
        "step_index": len(rows),
        "action": "stop",
        "budget_remaining_frac": round(1 - cum_before / budget, 4),
        "last_error": int(last_error),
        "url_changed_last": int(url_changed),
        "consecutive_errors": consecutive_errors,
        "step_cost": 0,
        "is_terminal": True,
        "termination_reason": ep.get("termination_reason"),
    })
    return rows

# mode pairs
def mode_pairs(episodes: list[dict]) -> list[dict]:
    #one row per (task, budget) with A's and B's outcome side by side
    cells: dict[tuple, dict] = defaultdict(dict)
    #group the episodes by task and budget
    for ep in episodes:
        #if the strategy is in ACTION_OF
        if ep["strategy"] in ACTION_OF:
            #add the episode to the cells
            cells[(ep["task_id"], ep["budget_level"])][ep["strategy"]] = ep
 
    #loop through the cells
    out = []
    for (tid, budget), arms in sorted(cells.items()):
        if "A" not in arms or "B" not in arms:
            continue
        a, b = arms["A"], arms["B"]
        row = {"task_id": tid, "budget_level": budget,
               **{f: a.get(f) for f in TASK_FIELDS},
               "tier_ord": TIER_ORD.get(a.get("difficulty_tier"))}
        for s, ep in (("a", a), ("b", b)):
            row[f"{s}_success"] = bool(ep["success"])
            row[f"{s}_tokens"] = ep["total_tokens"]
            row[f"{s}_eligible"] = ep.get("use_for_reward", True)
        row["both_eligible"] = row["a_eligible"] and row["b_eligible"]
        out.append(row)
    return out