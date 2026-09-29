"""
Since the AWS instance crashed sometimes or there were errors, re-running the "run_batch.py" resulted in some tasks/budgets being completed later.
i.e. 6 completed budgets for task 1, 
5 completed budgets for task 2, 
instance crashes & re-run batch script, 
6 completed budgets for task 3, 
1 completed budget for task 2.

Sorting will be completed by timestamp. If 1/6 budgets is completed later, then simply move it to the end of the list for that task.
"""

from __future__ import annotations
 
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path


def sort_episodes(episodes: list[dict]) -> list[dict]:
    by_task: dict[int, list[dict]] = defaultdict(list)
    for ep in episodes:
        by_task[ep.get("task_id")].append(ep)
 
    #Chronological within each task
    for eps in by_task.values():
        eps.sort(key=lambda e: e.get("timestamp", ""))
 
    #Blocks ordered by when the task first appeared
    ordered_tasks = sorted(
        by_task.keys(),
        key=lambda tid: by_task[tid][0].get("timestamp", ""),
    )
 
    return [ep for tid in ordered_tasks for ep in by_task[tid]]


def find_late_arrivals(episodes: list[dict], gap_minutes: int = 30) -> list[tuple]:

    from datetime import datetime
 
    by_task: dict[int, list[dict]] = defaultdict(list)
    for ep in episodes:
        by_task[ep.get("task_id")].append(ep)
 
    late = []
    for tid, eps in by_task.items():
        eps = sorted(eps, key=lambda e: e.get("timestamp", ""))
        for prev, cur in zip(eps, eps[1:]):
            try:
                t0 = datetime.fromisoformat(prev.get("timestamp", ""))
                t1 = datetime.fromisoformat(cur.get("timestamp", ""))
            except ValueError:
                continue
            gap = (t1 - t0).total_seconds() / 60
            if gap > gap_minutes:
                late.append((tid, cur.get("budget_level"),
                             cur.get("timestamp"), gap))
    return sorted(late, key=lambda x: -x[3])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True, help="collection JSONL")
    ap.add_argument("--out", default=None, help="output path (default: sort the file in place, keeping a .bak)")
    ap.add_argument("--gap-minutes", type=int, default=30, help="gap that marks an episode as a late arrival ""(default 30)")
    args = ap.parse_args()
 
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")
 
    # Read
    episodes, malformed, error_rows = [], 0, 0
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
            except json.JSONDecodeError:
                malformed += 1
                continue
            if "error" in ep:
                error_rows += 1
            episodes.append(ep)
 
    if not episodes:
        raise SystemExit(f"No episodes found in {path}")
 
    print(f"File: {path}")
    print(f"Episodes: {len(episodes)}")
    print(f"Tasks: {len({e.get('task_id') for e in episodes})}")

 
 
    # Report late arrivals 
    late = find_late_arrivals(episodes, args.gap_minutes)
    if late:
        print(f"\n{'=' * 70}")
        print(f"Late arrivals (>{args.gap_minutes} min after the task's "
              f"previous episode)")
        for tid, budget, ts, gap in late[:20]:
            hrs = gap / 60
            gap_str = f"{gap:.0f} min" if gap < 120 else f"{hrs:.1f} hrs"
            print(f"   task {tid:>4} @ {str(budget):>6}  {ts}  "
                  f"(+{gap_str})")
        if len(late) > 20:
            print(f"   ... and {len(late) - 20} more")
    else:
        print(f"\nNo gaps over {args.gap_minutes} minutes - the collection "
              f"appears to have run uninterrupted.")
 
    # Sort
    sorted_eps = sort_episodes(episodes)
 
    moved = sum(1 for a, b in zip(episodes, sorted_eps)
                if (a.get("task_id"), a.get("budget_level"))
                != (b.get("task_id"), b.get("budget_level")))
    print(f"\n{'=' * 70}")
    print(f"Rows changing position: {moved}/{len(episodes)}")
 
    # Show the first few blocks so the layout is visible
    print(f"\nFirst 3 task blocks after sorting:")
    shown_tasks, current = 0, None
    for ep in sorted_eps:
        tid = ep.get("task_id")
        if tid != current:
            if shown_tasks >= 3:
                break
            current = tid
            shown_tasks += 1
            print()
        print(f"task {tid:>4} @ {str(ep.get('budget_level')):>6}  "
              f"{ep.get('timestamp')}")
 
 
    # Write
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        out_path = path
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, backup)
        print(f"\nBackup written to {backup}")
 
    with out_path.open("w") as f:
        for ep in sorted_eps:
            f.write(json.dumps(ep) + "\n")
 
    print(f"Sorted {len(sorted_eps)} episodes into "
          f"{len({e.get('task_id') for e in sorted_eps})} task blocks")
    print(f"Written to {out_path}")
 
 
if __name__ == "__main__":
    main()