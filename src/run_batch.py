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
