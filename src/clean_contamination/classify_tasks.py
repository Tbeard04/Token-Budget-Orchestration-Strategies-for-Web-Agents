"""
classify_tasks.py - assign a contamination risk level to every task.

    read_only: Information retrieval, navigation, browsing. The agent reads but never writes. Running the same task again encounters an identical site, so every episode is clean no matter what ran before.

    ------------------------------------------------------------------------------------------------
    idempotent: Add to wishlist, add to cart, upvote, subscribe, rate. These DO change site state, 
    but repeating them does not change the outcome: an item already in the wishlist stays in the wishlist, an already
    upvoted post stays upvoted. The evaluator sees the correct final state whether the agent acted once or twice. Treated as clean.

    ------------------------------------------------------------------------------------------------
    non_idempotent: Change a price, delete reviews, post new content, update an address, cancel an order. A prior success leaves the site in a different
    starting condition, so a later episode may "succeed" by finding the work already done. Only these need contamination handling.

src/: python -m clean_contamination.classify_tasks \ --annotations ../data/processed/final_annotation_difficulty_tiers/task_metadata.jsonl \ --output ../data/processed/task_list/task_risk_levels.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

#risk levels
RISK_LEVELS = ["read_only", "idempotent", "non_idempotent"]

#Annotation categories that involve writing to the site
STATE_CHANGE_CATEGORIES = {"create", "modify_value", "delete", "bulk_action", "crud_operation", "form_fill",}

#Intent phrases indicating an idempotent write - repeating is harmless
IDEMPOTENT_KEYWORDS = ["add to wishlist", "wish list", "wishlist", "add to cart", "add to my cart", "to the shopping cart", "to my cart", "subscribe", "upvote", "up vote", "downvote", "down vote", "thumbs up", "thumbs down",]

#Intent phrases indicating the task only reads
ACTUALLY_READ_ONLY_KEYWORDS = ["show me", "tell me", "what is", "what are", "what do", "among the", "find the", "list out", "list the", "how many", "how much", "is there any", "compare", "count of", "which customer", "who has",]

#Patterns for writing verbs
WRITE_VERB_PATTERNS = [
    r"\breorder\b", r"\bre-order\b",
    r"\bbuy\b", r"\bpurchase\b", r"\bcheckout\b", r"\bplace an order\b",
    r"\bdelete\b", r"\bremove\b", r"\bcancel\b",
    r"\bchange\b", r"\bupdate\b", r"\bmodify\b", r"\bedit\b", r"\brename\b",
    r"\bset the\b", r"\bincrease\b", r"\breduce\b", r"\bdecrease\b",
    r"\bdisable\b", r"\benable\b", r"\bapprove\b",
    r"\bpost\b", r"\bcreate\b", r"\bdraft\b", r"\bsubmit\b", r"\breply\b",
    r"\bsubscribe\b", r"\bupvote\b", r"\bdownvote\b",
]

#check if the intent contains a phrase
def _has_phrase(intent: str, phrases: list[str]) -> bool:
    return any(p in intent for p in phrases)
 
#check if the intent contains a write verb
def _has_pattern(intent: str, patterns: list[str]) -> bool:
    return any(re.search(p, intent) for p in patterns)

#classify the task
def classify(intent: str, category: str) -> tuple[str, str]:
    #convert intent to lowercase
    i = (intent or "").lower()
    #if the category is a state change category
    if category in STATE_CHANGE_CATEGORIES:
        if _has_phrase(i, IDEMPOTENT_KEYWORDS):
            return "idempotent", f"{category}: idempotent action"
        return "non_idempotent", f"{category}: writes to the site"
    #if the intent contains a write verb
    if _has_pattern(i, WRITE_VERB_PATTERNS):
        if _has_phrase(i, IDEMPOTENT_KEYWORDS):
            return "idempotent", f"{category}: write verb, idempotent action"
        return "non_idempotent", f"{category}: write verb in the intent"
    #if the intent contains a read-only phrasing
    if _has_phrase(i, ACTUALLY_READ_ONLY_KEYWORDS):
        return "read_only", f"{category}: read-only phrasing"
    #if no write is detected
    return "read_only", f"{category}: no write detected"

#main
def main() -> None:
    ap = argparse.ArgumentParser()
    #add arguments
    ap.add_argument("--annotations", default="../data/processed/final_annotation_difficulty_tiers/task_metadata.jsonl")
    ap.add_argument("--output", default="../data/processed/task_list/task_risk_levels.jsonl")
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()
 
    #read the annotations
    src = Path(args.annotations)
 
    rows = []
    #list of tasks with unknown category
    unknown = []
    #read the annotations
    for line in src.read_text().splitlines():
        #strip the line
        line = line.strip()
        if not line:
            continue
        #load the task
        t = json.loads(line)
        #get the category
        cat = t.get("task_category", "unknown")
        #if the category is unknown
        if cat == "unknown":
            unknown.append(t["task_id"])
        #classify the task
        level, reason = classify(t.get("intent", ""), cat)
        #add the task to the list
        rows.append({
            #add the task id
            "task_id": t["task_id"],
            "site": t.get("site", ""),
            "intent": t.get("intent", ""),
            "task_category": cat,
            "risk_level": level,
            "risk_reason": reason,
        })
 
    #write the tasks to the output file
    out = Path(args.output)
    #create the output directory if it doesn't exist
    out.parent.mkdir(parents=True, exist_ok=True)
    #write the tasks to the output file
    with out.open("w") as f:
        #write each task
        for r in rows:
            f.write(json.dumps(r) + "\n")
 
    #count the number of tasks
    n = len(rows)
    #print the number of tasks read and written
    print(f"[classify] read  {n} tasks from {src}")
    print(f"[classify] wrote {n} tasks to  {out}")
 
    #if there are tasks with unknown category
    if unknown:
        #print the number of tasks with unknown category
        print(f"\n[classify] WARNING: {len(unknown)} tasks have "
              f"task_category 'unknown' and were classified on intent text "
              f"alone: {unknown[:15]}")
 
    #count the number of tasks by risk level
    counts = Counter(r["risk_level"] for r in rows)
    #print the risk levels
    print(f"\n{'=' * 70}\nRisk levels\n{'=' * 70}")
    for lv in RISK_LEVELS:
        c = counts.get(lv, 0)
        print(f"   {lv:16s} {c:>4}  ({c / n:>4.0%})  {'#' * int(40 * c / n)}")
 
    #count the number of clean tasks
    clean = counts.get("read_only", 0)
    #print the number of clean tasks
    print(f"\n {clean} tasks ({clean / n:.0%}) cannot be contaminated at all.")
    #print the number of tasks that need the first-success rule
    print(f"{n - clean} tasks ({1 - clean / n:.0%}) need the first-success rule.")
 
    #print the risk level by site
    print(f"\n{'=' * 70}\nRisk level by site\n{'=' * 70}")
    #count the number of tasks by site
    by_site = defaultdict(Counter)
    for r in rows:
        by_site[r["site"]][r["risk_level"]] += 1
    #print the risk level by site
    for site in sorted(by_site):
        c = by_site[site]
        print(f"   {site:16s} read_only:{c['read_only']:>4}  "
              f"idempotent:{c['idempotent']:>4}  "
              f"non_idempotent:{c['non_idempotent']:>4}")
 
    #print the category -> risk level
    print(f"\n{'=' * 70}\nCategory -> risk level\n{'=' * 70}")
    #count the number of tasks by category
    by_cat = defaultdict(Counter)
    for r in rows:
        by_cat[r["task_category"]][r["risk_level"]] += 1
    #print the category -> risk level
    for cat in sorted(by_cat, key=lambda c: -sum(by_cat[c].values())):
        c = by_cat[cat]
        parts = "  ".join(f"{lv}:{c[lv]}" for lv in RISK_LEVELS if c[lv])
        print(f"   {cat:22s} {parts}")
 
    if args.show:
        print(f"\n{'=' * 70}\nExamples\n{'=' * 70}")
        #print the examples
        for lv in RISK_LEVELS:
            #get the tasks with the risk level
            sub = [r for r in rows if r["risk_level"] == lv][:args.show]
            #print the risk level
            print(f"\n   --- {lv} ---")
            #print the tasks
            for r in sub:
                #print the task id and intent
                print(f"   {r['task_id']:>4}  {r['intent'][:62]}")
                #print the risk reason
                print(f"{r['risk_reason']}")