"""
annotate_tasks.py - entry point for difficulty annotation
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import tasks
from annotator import annotate_one, build_annotator, load_categories
from gold_set import (apply_gold, format_exemplars, load_gold_labels, split_exemplars, write_gold_set)
from observed_cost import load_observed_cost
from postprocess import flag_for_review, harmonise_templates, reband
from reporting import report
from rubric import TIERS, tier_of

#function to re-band the tasks
def do_reband(path: str) -> None:
    #load the rows from the path
    rows = [json.loads(l) for l in open(path) if l.strip()]
    #re-band the tasks
    changed = reband(rows)
    #write the rows to the path
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Re-banded {len(rows)} tasks, {changed} tiers changed")
    tiers = Counter(r["difficulty_tier"] for r in rows)
    #for each tier in the tiers
    for t in TIERS:
        #get the count for the tier
        c = tiers.get(t, 0)
        #print the tier and the count
        print(f"{t:8s} {c:>4}  ({c / max(len(rows), 1):.0%})")

#function to run the annotation
def run_annotation(args) -> None:
    #create the output path
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Calibration - some hand labels become exemplars, the rest are held back
    gold = load_gold_labels(args.gold_labels)
    exemplars, heldout = split_exemplars(gold, args.n_exemplars, args.seed)
    if gold:
        print(f"[annotate] exemplars shown to annotator: {len(exemplars)}  "
              f"held out for kappa: {len(heldout)}")

    #build the annotator
    agent = build_annotator(format_exemplars(exemplars))
    #load the categories
    categories = load_categories(args.intents)
    #load the observed cost
    observed = load_observed_cost(args.episodes)

    #sample the task ids
    task_ids = args.tasks or tasks.sample_task_ids(args.n, args.sites, args.seed)
    #load the configurations by id
    by_id = tasks.configs_by_id()

    print(f"[annotate] tasks: {len(task_ids)}")
    print(f"[annotate] output: {out_path}\n")

    #create a list to store the rows
    rows = []
    for i, tid in enumerate(task_ids, 1):
        #get the configuration by id
        cfg = by_id.get(tid)
        #if the configuration is not found
        if cfg is None:
            print(f"[annotate] {i}/{len(task_ids)} task {tid}: NOT FOUND")
            continue
        try:
            row = annotate_one(agent, cfg, categories.get(tid, "unknown"), observed.get(tid))
        except Exception as e:
            print(f"[annotate] {i}/{len(task_ids)} task {tid}: "
                  f"{type(e).__name__}: {e}")
            continue

        row["gold_exemplar"] = tid in exemplars
        row["gold_heldout"] = tid in heldout
        rows.append(row)

        #print the task id, site, rubric total, difficulty tier, mark, gmark, and task category
        mark = "*" if row.get("observed") else " "
        gmark = "E" if row["gold_exemplar"] else ("H" if row["gold_heldout"] else " ")
        print(f"[annotate] {i}/{len(task_ids)} task {tid:>4} "
              f"{row['site']:15s} {row['rubric_total']}/8 "
              f"{row['difficulty_tier']:6s} {mark}{gmark} {row['task_category']}")

    if not args.no_harmonise:
        adjusted = harmonise_templates(rows, protect=set(exemplars))
        print(f"\n[annotate] harmonised {adjusted} rows to their template median")

    #flag the rows for review
    flag_for_review(rows, args.review_threshold)

    #write the rows to the output path
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    report(rows, args.sites)
    print(f"\n[annotate] wrote {len(rows)} rows to {out_path}")
    if heldout:
        print(f"[annotate] next: python cohen_kappa.py --gold {args.gold_labels} "
              f"--pred {out_path}")

#function to main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", type=int)
    ap.add_argument("--n", type=int, default=999, help="tasks per site")
    ap.add_argument("--sites", nargs="+", default=tasks.SITES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--intents", default=None)
    ap.add_argument("--episodes", nargs="+", default=None)
    ap.add_argument("--gold-labels", default=None)
    ap.add_argument("--n-exemplars", type=int, default=20)
    ap.add_argument("--review-threshold", type=float, default=0.75)
    ap.add_argument("--no-harmonise", action="store_true")
    ap.add_argument("--out", default="../../data/tasks/task_metadata.jsonl")
    ap.add_argument("--gold-out", default="../../data/raw/annotation_tests/run2/gold_set_50.jsonl")
    ap.add_argument("--reband", default=None)
    ap.add_argument("--gold-set", type=int, default=None)
    ap.add_argument("--apply-gold", default=None)
    args = ap.parse_args()

    #if the reband argument is provided
    if args.reband:
        #re-band the tasks
        do_reband(args.reband)
        return

    #if the gold set argument is provided
    if args.gold_set:
        #if the gold out path is the same as the output path
        if Path(args.gold_out) == Path(args.out):
            ap.error("--gold-out must differ from --out")
        #write the gold set
        write_gold_set(args.gold_set, args.gold_out, args.seed, args.sites, args.episodes)
        return

    #if the apply gold argument is provided
    if args.apply_gold:
        #load the gold labels
        gold = load_gold_labels(args.gold_labels)
        if not gold:
            ap.error("--apply-gold needs a --gold-labels file with completed scores")
        apply_gold(args.apply_gold, gold, args.out)
        return

    #run the annotation
    run_annotation(args)

if __name__ == "__main__":
    main()