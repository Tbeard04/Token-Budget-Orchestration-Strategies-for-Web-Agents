"""
annotate_tasks.py - entry point for difficulty annotation

Run from src/task_annotation/, in that order. Steps 1 and 4 make no LLM calls.

 1. Write the blank gold set, then fill in the four scores by hand:
      python annotate_tasks.py --gold-set 50 \
        --episodes ../../data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_A/strategy_a.jsonl \
        --gold-out ../../data/raw/annotation_tests/run2/gold_set_50.jsonl

 2. Annotate all tasks, anchored on the hand labels (475 LLM calls - use tmux):
      python annotate_tasks.py --n 999 \
        --intents ../../data/tasks/task_intents.json \
        --episodes ../../data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_A/strategy_a.jsonl \
        --gold-labels ../../data/raw/annotation_tests/run2/gold_set_50.jsonl \
        --out ../../data/raw/annotation_tests/run2/tasks_annotated_run2.jsonl

 3. Agreement, BEFORE any merge:
      python cohen_kappa.py \
        --gold ../../data/raw/annotation_tests/run2/gold_set_50.jsonl \
        --pred ../../data/raw/annotation_tests/run2/tasks_annotated_run2.jsonl \
        --out  ../../data/raw/annotation_tests/run2/kappa_run2.json

 4. Final metadata, with the 50 human labels replacing the model's:
      python annotate_tasks.py \
        --apply-gold ../../data/raw/annotation_tests/run2/tasks_annotated_run2.jsonl \
        --gold-labels ../../data/raw/annotation_tests/run2/gold_set_50.jsonl \
        --out ../../data/tasks/task_metadata.jsonl
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


def do_reband(path: str) -> None:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    changed = reband(rows)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Re-banded {len(rows)} tasks, {changed} tiers changed")
    tiers = Counter(r["difficulty_tier"] for r in rows)
    for t in TIERS:
        c = tiers.get(t, 0)
        print(f"{t:8s} {c:>4}  ({c / max(len(rows), 1):.0%})")


def run_annotation(args) -> None:
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Calibration - some hand labels become exemplars, the rest are held back
    gold = load_gold_labels(args.gold_labels)
    exemplars, heldout = split_exemplars(gold, args.n_exemplars, args.seed)
    if gold:
        print(f"[annotate] exemplars shown to annotator: {len(exemplars)}  "
              f"held out for kappa: {len(heldout)}")

    agent = build_annotator(format_exemplars(exemplars))
    categories = load_categories(args.intents)
    observed = load_observed_cost(args.episodes)

    task_ids = args.tasks or tasks.sample_task_ids(args.n, args.sites, args.seed)
    by_id = tasks.configs_by_id()

    print(f"[annotate] tasks: {len(task_ids)}")
    print(f"[annotate] output: {out_path}\n")

    rows = []
    for i, tid in enumerate(task_ids, 1):
        cfg = by_id.get(tid)
        if cfg is None:
            print(f"[annotate] {i}/{len(task_ids)} task {tid}: NOT FOUND")
            continue
        try:
            row = annotate_one(agent, cfg, categories.get(tid, "unknown"),
                               observed.get(tid))
        except Exception as e:
            print(f"[annotate] {i}/{len(task_ids)} task {tid}: "
                  f"{type(e).__name__}: {e}")
            continue

        row["gold_exemplar"] = tid in exemplars
        row["gold_heldout"] = tid in heldout
        rows.append(row)

        mark = "*" if row.get("observed") else " "
        gmark = "E" if row["gold_exemplar"] else ("H" if row["gold_heldout"] else " ")
        print(f"[annotate] {i}/{len(task_ids)} task {tid:>4} "
              f"{row['site']:15s} {row['rubric_total']}/8 "
              f"{row['difficulty_tier']:6s} {mark}{gmark} {row['task_category']}")

    if not args.no_harmonise:
        adjusted = harmonise_templates(rows, protect=set(exemplars))
        print(f"\n[annotate] harmonised {adjusted} rows to their template median")

    flag_for_review(rows, args.review_threshold)

    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    report(rows, args.sites)
    print(f"\n[annotate] wrote {len(rows)} rows to {out_path}")
    if heldout:
        print(f"[annotate] next: python cohen_kappa.py --gold {args.gold_labels} "
              f"--pred {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", type=int,
                    help="explicit task ids; otherwise sampled like run_batch")
    ap.add_argument("--n", type=int, default=999, help="tasks per site")
    ap.add_argument("--sites", nargs="+", default=tasks.SITES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--intents", default=None,
                    help="task_intents.json from extract_task_intents.py")
    ap.add_argument("--episodes", nargs="+", default=None,
                    help="one or more collection files; list Strategy A FIRST")
    ap.add_argument("--gold-labels", default=None,
                    help="hand-labelled gold set; used as few-shot exemplars")
    ap.add_argument("--n-exemplars", type=int, default=20,
                    help="how many gold tasks to show the annotator; the rest "
                         "are held out for an honest kappa")
    ap.add_argument("--review-threshold", type=float, default=0.75)
    ap.add_argument("--no-harmonise", action="store_true",
                    help="skip template harmonisation (not recommended)")
    ap.add_argument("--out", default="../../data/tasks/task_metadata.jsonl")
    ap.add_argument("--gold-out",
                    default="../../data/raw/annotation_tests/run2/gold_set_50.jsonl",
                    help="where --gold-set writes; never the same file as --out")
    ap.add_argument("--reband", default=None,
                    help="re-derive tiers in an existing file and exit")
    ap.add_argument("--gold-set", type=int, default=None,
                    help="write N blank tasks for hand-labelling and exit")
    ap.add_argument("--apply-gold", default=None,
                    help="annotated file to merge --gold-labels into, then exit")
    args = ap.parse_args()

    if args.reband:
        do_reband(args.reband)
        return

    if args.gold_set:
        if Path(args.gold_out) == Path(args.out):
            ap.error("--gold-out must differ from --out")
        write_gold_set(args.gold_set, args.gold_out, args.seed, args.sites, args.episodes)
        return

    if args.apply_gold:
        gold = load_gold_labels(args.gold_labels)
        if not gold:
            ap.error("--apply-gold needs a --gold-labels file with completed scores")
        apply_gold(args.apply_gold, gold, args.out)
        return

    run_annotation(args)


if __name__ == "__main__":
    main()