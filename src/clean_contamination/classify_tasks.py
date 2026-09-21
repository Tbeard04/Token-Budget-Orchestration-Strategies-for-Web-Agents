"""
classify_tasks.py - assign a contamination risk level to every task.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

#List of risk levels
RISK_LEVELS = ["read_only", "idempotent", "non_idempotent"]

#List of categories that write to the site
STATE_CHANGE_CATEGORIES = {"create", "modify_value", "delete", "bulk_action", "purchase","crud_operation", "form_fill",}

#List of keywords that indicate an idempotent write
IDEMPOTENT_KEYWORDS = ["add to wishlist", "wish list", "wishlist", "add to cart", "add to my cart", "to the shopping cart", "to my cart", "subscribe", "upvote", "up vote", "downvote", "down vote", "thumbs up", "thumbs down",]

#Intent phrases indicating the task only reads
ACTUALLY_READ_ONLY_KEYWORDS = ["show me", "tell me", "what is", "what are", "what do", "among the", "find the", "list out", "list the", "how many", "how much", "is there any", "compare", "count of", "which customer", "who has", "get the", "give me", "provide me", "summarize", "summarise",]

#List of patterns that indicate a strong write
STRONG_WRITE_PATTERNS = [
    r"\breorder\b", r"\bre-order\b",
    r"\bbuy\b", r"\bcheckout\b", r"\bplace an order\b",
    r"\bdelete\b", r"\bremove\b", r"\bcancel\b",
    r"\bchange\b", r"\bupdate\b", r"\bmodify\b", r"\bedit\b", r"\brename\b",
    r"\bset the\b", r"\bincrease\b", r"\breduce\b", r"\bdecrease\b",
    r"\bdisable\b", r"\benable\b", r"\bapprove\b",
    r"\bcreate\b", r"\bdraft\b", r"\bsubmit\b", r"\breply\b",
    r"\bsubscribe\b", r"\bnotify\b",
]

#List of patterns that indicate an ambiguous write
AMBIGUOUS_WRITE_PATTERNS = [
    r"\bpost\b", r"\bposting\b", r"\bpurchase\b",
    r"\bupvote\b", r"\bdownvote\b", r"\brate\b",
]


#Function to strip quoted spans before looking for instruction words
def _strip_quoted(intent: str) -> str:
    return re.sub(r"'[^']*'", " ", re.sub(r'"[^"]*"', " ", intent))


#Function to check if the intent contains any of the phrases
def _has_phrase(intent: str, phrases: list[str]) -> bool:
    return any(p in intent for p in phrases)


#Function to check if the intent contains any of the patterns
def _has_pattern(intent: str, patterns: list[str]) -> bool:
    return any(re.search(p, intent) for p in patterns)


def is_impossible(cfg: dict) -> bool:
    #get the reference answers
    ans = (cfg.get("eval") or {}).get("reference_answers")
    #if the reference answers is not a dictionary
    if not isinstance(ans, dict):
        #return False
        return False
    values: list[str] = []
    for key in ("exact_match", "must_include", "fuzzy_match"):
        v = ans.get(key)
        if isinstance(v, str):
            values.append(v)
        elif isinstance(v, list):
            values.extend(str(x) for x in v)
    #every accepted answer must be N/A - a task that merely allows N/A among other answers is a real task
    return bool(values) and all(v.strip().upper() == "N/A" for v in values)
 
 
def load_impossible(configs_path: str | None) -> set:
    #if the configs path is not provided
    if not configs_path:
        return set()
    #get the path
    p = Path(configs_path)
    #if the path does not exist
    if not p.exists():
        #print the error
        print(f"[classify] configs not found, impossible-task check skipped: {p}")
        return set()
    #load the data
    data = json.loads(p.read_text())
    #get the task ids
    ids = {c["task_id"] for c in data if "task_id" in c and is_impossible(c)}
    #print the number of impossible tasks
    print(f"[classify] impossible tasks (evaluator expects N/A): {len(ids)}"
          + (f"  {sorted(ids)[:15]}" if ids else ""))
    return ids

#Function to classify the intent
def classify(intent: str, category: str, impossible: bool = False) -> tuple[str, str]:
    #convert the intent to lowercase
    i = (intent or "").lower()
    #strip the quoted spans
    bare = _strip_quoted(i)
    #if the category is a state change category


    #an impossible task writes nothing whatever its category says
    if impossible:
        return "read_only", f"{category}: impossible task, evaluator expects N/A"

    if category in STATE_CHANGE_CATEGORIES:
        if (not _has_pattern(bare, STRONG_WRITE_PATTERNS)
                and _has_phrase(bare, ACTUALLY_READ_ONLY_KEYWORDS)
                and not _has_phrase(i, IDEMPOTENT_KEYWORDS)):
            return "read_only", f"{category}: read-only phrasing, no write verb"
        #if the intent contains any of the idempotent keywords
        if _has_phrase(i, IDEMPOTENT_KEYWORDS):
            return "idempotent", f"{category}: idempotent action"
        return "non_idempotent", f"{category}: writes to the site"

    #if the intent contains any of the strong write patterns
    if _has_pattern(bare, STRONG_WRITE_PATTERNS):
        #if the intent contains any of the idempotent keywords
        if _has_phrase(i, IDEMPOTENT_KEYWORDS):
            return "idempotent", f"{category}: write verb, idempotent action"
        return "non_idempotent", f"{category}: write verb in the intent"

    #if the intent contains any of the actually read only keywords
    if _has_phrase(bare, ACTUALLY_READ_ONLY_KEYWORDS):
        return "read_only", f"{category}: read-only phrasing"

    #if the intent contains any of the ambiguous write patterns
    if _has_pattern(bare, AMBIGUOUS_WRITE_PATTERNS):
        if _has_phrase(i, IDEMPOTENT_KEYWORDS):
            return "idempotent", f"{category}: write verb, idempotent action"
        return "non_idempotent", f"{category}: write verb in the intent"

    return "read_only", f"{category}: no write detected"


#main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--annotations", default="../data/processed/final_annotation_difficulty_tiers/task_metadata.jsonl")
    ap.add_argument("--output", default="../data/processed/task_list/task_risk_levels.jsonl")
    ap.add_argument("--configs", default="task_analysis/test.raw.json")
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    #self test
    if args.self_test:
        self_test()
        return

    #load the tasks
    src = Path(args.annotations)
    if not src.exists():
        raise SystemExit(f"not found: {src}\n"
                         f"Run annotate_tasks.py --apply-gold first.")

    #load the impossible tasks
    impossible = load_impossible(args.configs)

    #list to store the tasks
    rows = []
    #list to store the tasks with unknown category
    unknown = []
    #iterate over the tasks
    for line in src.read_text().splitlines():
        #strip the line
        line = line.strip()
        #if the line is empty
        if not line:
            continue
        #load the task
        t = json.loads(line)
        #get the category
        cat = t.get("task_category", "unknown")
        #if the category is unknown
        if cat == "unknown":
            #add the task id to the list
            unknown.append(t["task_id"])
        level, reason = classify(t.get("intent", ""), cat, t["task_id"] in impossible)
        #add the task to the list
        rows.append({
            "task_id": t["task_id"],
            "site": t.get("site", ""),
            "intent": t.get("intent", ""),
            "task_category": cat,
            "risk_level": level,
            "risk_reason": reason,
        })

    #save the tasks
    out = Path(args.output)
    #create the parent directory if it doesn't exist
    out.parent.mkdir(parents=True, exist_ok=True)
    #write the tasks to the file
    with out.open("w") as f:
        #write each task to the file
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
    print(f"\n   {clean} tasks ({clean / n:.0%}) cannot be contaminated at all.")
    #print the number of tasks that need the first-success rule
    print(f"   {n - clean} tasks ({1 - clean / n:.0%}) need the first-success rule.")

    #print the risk level by site
    print(f"\n{'=' * 70}\nRisk level by site\n{'=' * 70}")
    by_site = defaultdict(Counter)
    for r in rows:
        by_site[r["site"]][r["risk_level"]] += 1
    for site in sorted(by_site):
        c = by_site[site]
        print(f"   {site:16s} read_only:{c['read_only']:>4}  "
              f"idempotent:{c['idempotent']:>4}  "
              f"non_idempotent:{c['non_idempotent']:>4}")

    #print the category -> risk level
    print(f"\n{'=' * 70}\nCategory -> risk level\n{'=' * 70}")
    by_cat = defaultdict(Counter)
    for r in rows:
        by_cat[r["task_category"]][r["risk_level"]] += 1
    for cat in sorted(by_cat, key=lambda c: -sum(by_cat[c].values())):
        c = by_cat[cat]
        parts = "  ".join(f"{lv}:{c[lv]}" for lv in RISK_LEVELS if c[lv])
        print(f"   {cat:22s} {parts}")

    #print the examples
    if args.show:
        print(f"\n{'=' * 70}\nExamples\n{'=' * 70}")
        for lv in RISK_LEVELS:
            sub = [r for r in rows if r["risk_level"] == lv][:args.show]
            print(f"\n   --- {lv} ---")
            for r in sub:
                print(f"   {r['task_id']:>4}  {r['intent'][:62]}")
                print(f"         {r['risk_reason']}")



# Self-test
def self_test() -> None:
    #list of cases to test
    cases = [
        # (intent, category, expected)
        ("What is the top-1 best-selling brand in Quarter 1 2022",
         "information_retrieval", "read_only"),
        ("I want to browse the products in the Headphones category",
         "navigation", "read_only"),
        ("List products from PS4 accessories category by ascending price",
         "other", "read_only"),
        ("Get the order number of my most recent complete order",
         "other", "read_only"),
        ("Show me products under $25 in \"women shoes\" category",
         "navigation", "read_only"),

        ("Add this product to my wishlist", "create", "idempotent"),
        ("Add Tide PODS to my wish list", "create", "idempotent"),
        ("Subscribe to the newsletter of OneStopMarket", "create", "idempotent"),
        ("Upvote the newest post in books subreddit", "bulk_action", "idempotent"),
        ("Thumbs down the top 1 post ever in gadgets.", "bulk_action", "idempotent"),

        # the two keyword changes
        ("Like all submissions created by Hrekires in subreddit news",
         "bulk_action", "non_idempotent"),
        ("DisLike all submissions created by RickyDontLoseThat in subreddit massachusetts",
         "bulk_action", "non_idempotent"),
        ("Rate my recent purchase of Jiffy Corn Muffin Cornbread Mix with 4 stars",
         "purchase", "non_idempotent"),

        ("Reduce the price of this product by 15%", "modify_value", "non_idempotent"),
        ("Delete all pending negative reviews for Circe fleece", "delete", "non_idempotent"),
        ("Change my reddit bio to \"I am a robot\"", "modify_value", "non_idempotent"),
        ("Post my question, \"is car necessary in NYC\", in a subreddit",
         "create", "non_idempotent"),
        ("Add a simple product named Lelelumon Yoga Mat with 42 in stock",
         "create", "non_idempotent"),
        ("Buy the highest rated product from the meat substitute category",
         "purchase", "non_idempotent"),
        ("Cancel order 302", "modify_value", "non_idempotent"),
        ("Disable Ryker Tee Crew Neck from the site", "modify_value", "non_idempotent"),

        # write verb hiding in the "other" fallback bucket
        ("I previously ordered some a mattress foundation around Feb or March "
         "2023 and later cancelled. Can you reorder it for me?",
         "other", "non_idempotent"),

        # word boundary: "remover" must not match "remove"
        ("Show me products under $46.99 in makeup remover",
         "information_retrieval", "read_only"),

        # noun/verb homographs: these only READ, despite containing post /
        # purchase / downvote (tasks 66, 117, 27)
        ("Among the top 10 post in \"books\" forum, show me the post URLs "
         "that recommand a single book", "information_retrieval", "read_only"),
        ("What is the date when I made my first purchase on this site?",
         "information_retrieval", "read_only"),
        ("Tell me the count of comments that have received more downvotes "
         "than upvotes for the user who made the latest post on the space "
         "forum.", "information_retrieval", "read_only"),

        # the mirror case: quoted text is payload, so 'what is' inside the
        # question being posted must NOT make this read-only (tasks 600, 609)
        ("Post my question, \"what is the recommended console to buy these "
         "days\", in a subreddit where I'm likely to get an answer",
         "information_retrieval", "non_idempotent"),
        ("Find a subreddit focused on topics related to ML, DL, NLP, and post "
         "my question, \"what is the SOTA web navigation agent repo\" there",
         "information_retrieval", "non_idempotent"),
        ("Find a subreddit focused on topics related to NYC, and post my "
         "question, \"is car necessary\" there", "navigation", "non_idempotent"),


        #homograph at the CATEGORY level
        ("Get the purchase date and order id of the most recent pending order",
         "purchase", "read_only"),
        ("Draft a new marketing price rule for fall discount that offers $10 "
         "discount on checkout for all customers", "purchase", "non_idempotent"),
        ("Add the product with the lowest per unit price from my open tabs "
         "to the shopping cart", "create", "idempotent"),
    ]

    #list of failures
    failures = []
    #test each case
    for intent, cat, expected in cases:
        got, reason = classify(intent, cat)
        if got != expected:
            failures.append((intent[:55], cat, expected, got, reason))


     # impossible tasks override the category (tasks 794-798)
    imp_cases = [
        ("Change the delivery address for my most recent order to 4000 Forbes "
         "Ave, Pittsburgh, PA.", "modify_value", "read_only"),
        ("Delete all pending negative reviews for Circe fleece", "delete",
         "non_idempotent"),   # not impossible: still non_idempotent
    ]
    for intent, cat, expected in imp_cases[:1]:
        got, _ = classify(intent, cat, impossible=True)
        if got != expected:
            failures.append((intent[:55], cat, expected, got, "impossible=True"))
    for intent, cat, expected in imp_cases[1:]:
        got, _ = classify(intent, cat, impossible=False)
        if got != expected:
            failures.append((intent[:55], cat, expected, got, "impossible=False"))
 
    # the N/A detector itself
    detector_cases = [
        ({"eval": {"reference_answers": {"fuzzy_match": "N/A"}}}, True),
        ({"eval": {"reference_answers": {"fuzzy_match": ["N/A"]}}}, True),
        ({"eval": {"reference_answers": {"exact_match": "n/a"}}}, True),
        ({"eval": {"reference_answers": {"must_include": ["N/A", "Sprite"]}}}, False),
        ({"eval": {"reference_answers": {"exact_match": "Sprite"}}}, False),
        ({"eval": {"reference_answers": None}}, False),
        ({"eval": {"eval_types": ["program_html"], "program_html": [{}]}}, False),
        ({}, False),
    ]

    #test the detector cases
    for cfg, expected in detector_cases:
        if is_impossible(cfg) != expected:
            failures.append((str(cfg)[:55], "is_impossible", expected, is_impossible(cfg), ""))
    cases = cases + imp_cases + detector_cases


    #print the failures
    for f in failures:
        #print the failure
        print(f"FAIL  {f[0]!r} [{f[1]}]\n      expected {f[2]}, got {f[3]}  ({f[4]})")

    #print the number of cases passed
    print(f"\nself-test: {len(cases) - len(failures)}/{len(cases)} passed")
    #if there are failures
    if failures:
        raise SystemExit(1)

if __name__ == "__main__":
    main()