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
    """Group into per-task blocks; order blocks by each task's first
    timestamp; keep episodes chronological within a block."""
    by_task: dict[int, list[dict]] = defaultdict(list)
    for ep in episodes:
        by_task[ep.get("task_id")].append(ep)
 
    # Chronological within each task
    for eps in by_task.values():
        eps.sort(key=lambda e: e.get("timestamp", ""))
 
    # Blocks ordered by when the task first appeared
    ordered_tasks = sorted(
        by_task.keys(),
        key=lambda tid: by_task[tid][0].get("timestamp", ""),
    )
 
    return [ep for tid in ordered_tasks for ep in by_task[tid]]