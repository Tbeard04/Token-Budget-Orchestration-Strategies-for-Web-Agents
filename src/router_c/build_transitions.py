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

#if the joined file dropped step_log, pull it back from the raw files (in this case I haven't dropped it)
def attach_step_logs(episodes: list[dict], raw_paths: list[str]) -> int:
    #if the episodes have step_log, return 0
    if episodes and "step_log" in episodes[0]:
        #return 0
        return 0
    #create a dictionary to store the step logs
    raw = {}
    #loop through the raw paths
    for p in raw_paths:
        for r in load_jsonl(p):
            if "error" not in r:
                raw[(r["strategy"], r["task_id"], r["budget_level"])] = r.get("step_log", [])
    n = 0
    for ep in episodes:
        ep["step_log"] = raw.get((ep["strategy"], ep["task_id"], ep["budget_level"]), [])
        n += 1
    return n

def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

#main function
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="../data/processed/6_budgets_ALL_tasks_decontaminated_batch/final_episodes.jsonl")
    ap.add_argument("--raw", nargs="*", default=[
        "../data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_A/strategy_a.jsonl",
        "../data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_B/strategy_b.jsonl",])
    ap.add_argument("--out-dir", default="../data/processed/router_c_models")
    args = ap.parse_args()

    #load the episodes
    episodes = load_jsonl(args.episodes)
    #attach the step logs
    restored = attach_step_logs(episodes, args.raw)
    print(f"[build] {len(episodes)} episodes"
          + (f", step_log restored from raw for {restored}" if restored else ""))

    #create the transitions
    trans = [r for ep in episodes for r in episode_rows(ep)]
    #create the mode pairs
    pairs = mode_pairs(episodes)

    #write the transitions and mode pairs to the output directory
    out = Path(args.out_dir)
    #write the transitions to the output directory
    write_jsonl(trans, out / "transitions.jsonl")
    #write the mode pairs to the output directory
    write_jsonl(pairs, out / "mode_pairs.jsonl")

    #count the actions
    acts = Counter(r["action"] for r in trans)
    print(f"[build] transitions {len(trans):>6}  "
          + "  ".join(f"{a} {n}" for a, n in sorted(acts.items())))
    print(f"[build] mode pairs  {len(pairs):>6}  "
          f"both eligible {sum(p['both_eligible'] for p in pairs)}")
    empty = sum(1 for ep in episodes if not ep.get("step_log"))
    if empty:
        print(f"[build] {empty} episodes had no steps -> stop row only")
    print(f"[build] wrote -> {out}/")


if __name__ == "__main__":
    main()