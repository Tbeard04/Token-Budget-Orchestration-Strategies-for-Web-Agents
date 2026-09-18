from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path


#Aggregate cost per task_id across one or more collection files
def load_observed_cost(episode_paths) -> dict:
    if not episode_paths:
        return {}
    if isinstance(episode_paths, (str, Path)):
        episode_paths = [episode_paths]

    observed: dict = {}
    for path in episode_paths:
        if not Path(path).exists():
            print(f"[observed] not found, skipped: {path}")
            continue

        by_task = defaultdict(list)
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    ep = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "error" in ep:
                    continue
                by_task[ep.get("task_id")].append(ep)

        added = 0
        for tid, eps in by_task.items():
            acting = [e for e in eps if e.get("steps", 0) > 0]
            if not acting:
                continue

            urls = set()
            for e in acting:
                for s in e.get("step_log") or []:
                    if s.get("url"):
                        urls.add(s["url"])
                    if s.get("next_url"):
                        urls.add(s["next_url"])

            stat = {
                "median_steps": statistics.median(e["steps"] for e in acting),
                "max_steps": max(e["steps"] for e in acting),
                "median_tokens": statistics.median(e.get("total_tokens", 0) for e in acting),
                "distinct_urls": len(urls),
                "episodes": len(acting),
                "source": Path(path).stem,
            }
            # first file wins: A's numbers are never overwritten by B's
            if observed.setdefault(tid, stat) is stat:
                added += 1

        print(f"[observed] +{added} tasks from {Path(path).name}")

    if observed:
        print(f"[observed] cost loaded for {len(observed)} tasks total")
    return observed


def format_observed(obs) -> str:
    #Render one task's cost for the annotator prompt. Empty when unknown
    if not obs:
        return ""
    return (
        f"\nObserved cost (a baseline single agent, across budget levels):\n"
        f"median steps taken: {obs['median_steps']:.0f}\n"
        f"maximum steps taken: {obs['max_steps']}\n"
        f"distinct pages visited: {obs['distinct_urls']}\n"
        f"median tokens: {obs['median_tokens']:.0f}\n"
        f"Use this to calibrate PAGES TO TRAVERSE only. The agent may have "
        f"wandered, so distinct pages is an upper bound."
    )


def summarise_hint(obs) -> dict:
    #The subset shown on a hand-labelling row
    return {"median_steps": obs["median_steps"], "max_steps": obs["max_steps"], "distinct_urls": obs["distinct_urls"],}