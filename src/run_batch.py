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
from itertools import groupby

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


def load_completed(path: Path) -> set[tuple]:
    """Read an existing results file and return the finished episodes.

    Tolerates a truncated final line, which is what a hard interrupt leaves.
    """
    done: set[tuple] = set()
    if not path.exists():
        return done
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "task_id" in r and "budget_level" in r:
                done.add((r.get("strategy"), r["task_id"], r["budget_level"]))
    return done

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", choices=["A", "B"], required=True)
    ap.add_argument("--n", type=int, default=67, help="tasks per site")
    ap.add_argument("--sites", nargs="+", default=W.SITES)
    ap.add_argument("--budgets", nargs="+", type=int, default=W.BUDGETS)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    out_path = Path(
        args.out or f"../data/raw/strategy_{args.strategy.lower()}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    runner = (strategy_a.run_episode if args.strategy == "A"
              else strategy_b.run_episode)

    plan = sample_tasks(args.n, args.sites, args.seed)
    done = load_completed(out_path)

    todo = [(site, tid, b)
            for site, ids in plan.items()
            for tid in ids
            for b in args.budgets
            if (args.strategy, tid, b) not in done]

    # Group by (site, task_id), shuffle budgets within each group, then shuffle the group order. This keeps a task's budget conditions adjacent while still spreading order effects.
    todo.sort(key=lambda x: (x[0], x[1]))
    rng = random.Random(args.seed)
    groups = [list(g) for _, g in groupby(todo, key=lambda x: (x[0], x[1]))]
    for g in groups:
        rng.shuffle(g)
    rng.shuffle(groups)
    todo = [item for g in groups for item in g]

    planned = sum(len(ids) for ids in plan.values()) * len(args.budgets)
    print(f"\n[batch] strategy    : {args.strategy}")
    print(f"[batch] sites       : {args.sites}")
    print(f"[batch] budgets     : {args.budgets}")
    print(f"[batch] seed        : {args.seed}")
    print(f"[batch] output      : {out_path}")
    print(f"[batch] planned     : {planned} episodes")
    print(f"[batch] already done: {len(done)}")
    print(f"[batch] to run      : {len(todo)}")
    for site, ids in plan.items():
        print(f"[batch]   {site:16s} {len(ids)} tasks")

    if not todo:
        print("\n[batch] nothing to do - all episodes already collected")
        return

    t_start = time.time()
    n_ok = n_err = n_success = 0

    # Append: never truncate an existing results file.
    with out_path.open("a") as fh:
        for idx, (site, tid, budget) in enumerate(todo, 1):
            elapsed = time.time() - t_start
            rate = elapsed / max(idx - 1, 1)
            eta_min = (len(todo) - idx + 1) * rate / 60
            print(f"\n[batch] {idx}/{len(todo)}  {site} task {tid} @ {budget}"
                  f"   (elapsed {elapsed/60:.0f}m, eta {eta_min:.0f}m,"
                  f" ok {n_ok}, success {n_success}, err {n_err})")

            try:
                rec = runner(tid, budget)
                n_ok += 1
                if rec.get("success"):
                    n_success += 1
            except KeyboardInterrupt:
                print("\n[batch] interrupted - completed episodes are saved.")
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                n_err += 1
                # Record the failure so a resume does not retry it forever.
                rec = {
                    "strategy": args.strategy,
                    "site": site,
                    "task_id": tid,
                    "budget_level": budget,
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "error": f"{type(e).__name__}: {e}",
                }

            # Write immediately: an interrupt must never lose finished work.
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    mins = (time.time() - t_start) / 60
    print(f"\n{'=' * 78}")
    print(f"[batch] finished in {mins:.0f} minutes")
    print(f"[batch]   episodes ok : {n_ok}")
    print(f"[batch]   successes   : {n_success}"
          f"  ({n_success / max(n_ok, 1):.1%})")
    print(f"[batch]   errors      : {n_err}")
    print(f"[batch] results in {out_path}")


if __name__ == "__main__":
    main()