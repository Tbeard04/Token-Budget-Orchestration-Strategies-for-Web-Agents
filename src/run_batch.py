"""
run_batch.py - full data collection with checkpointing and resume.

Runs a strategy across a task subset and all budget levels, writing every
episode to JSONL immediately and tracking completed (strategy, task, budget)
triples so an interrupted run resumes instead of restarting.

Designed to run unattended under tmux:
    tmux new -s collect
    python run_batch.py --strategy A --sites reddit --n 67
    # detach: Ctrl-b then d       reattach: tmux attach -t collect

Recommended sequence:
    # 1. pilot: validate the schema
    python run_batch.py --strategy A --n 3 --out ../data/raw/pilot_a.jsonl

    # 2. real collection, one site at a time (only one is hosted at once)
    python run_batch.py --strategy A --sites reddit --n 67
    python run_batch.py --strategy A --sites shopping --n 67
    python run_batch.py --strategy A --sites shopping_admin --n 66

    # 3. after a crash, rerun the identical command - finished episodes skip
    python run_batch.py --strategy A --sites reddit --n 67

Budget order is shuffled with a fixed seed. State-change tasks ("update my
address", "delete all reviews") alter the site, so a task run at several budget
levels may be contaminated by its own earlier runs. This means resetting the sites to its original state before re-running this script.
Shuffling spreads any such order effect evenly across conditions rather than letting it always favour the
last budget run.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import wa_env as W
import strategy_a
import strategy_b


def sample_tasks(n: int, sites: list[str], seed: int) -> dict[str, list[int]]:
    """Randomly sample n single-site task ids per site, reproducibly.

    Random rather than first-n: WebArena tasks are template generated, so
    consecutive ids are often variants of the same question and would give an
    unrepresentative subset.
    """
    pools = W.single_site_tasks(sites)
    rng = random.Random(seed)
    out: dict[str, list[int]] = {}
    for site in sites:
        pool = pools.get(site, [])
        if not pool:
            print(f"[batch] WARNING: no single-site tasks found for '{site}'")
            out[site] = []
            continue
        k = min(n, len(pool))
        if k < n:
            print(f"[batch] note: only {k} tasks available for '{site}'")
        out[site] = sorted(rng.sample(pool, k))
    return out
