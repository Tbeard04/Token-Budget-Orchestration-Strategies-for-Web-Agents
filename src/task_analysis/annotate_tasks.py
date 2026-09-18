from __future__ import annotations
 
import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
 
from pydantic import BaseModel, Field
 
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wa_env as W

#Rubric output
#The LLM scores four dimensions and nothing else. Category comes from extract_task_intents.py, which is regex-based, self-tested and consistent.
 
class TaskAnnotation(BaseModel):
    pages_to_traverse: int = Field(ge=0, le=2, description="0 single page, 1 two-to-three pages, 2 four-plus or unbounded")
    retrieval_type: int = Field(ge=0, le=2, description="0 read one value, 1 compare or filter a few, 2 aggregate or count over a set")
    interaction: int = Field(ge=0, le=2, description="0 read-only, 1 one form or click sequence, 2 multi-step state change")
    target_locatability: int = Field(ge=0, le=2, description="0 target named explicitly, 1 derivable from the page, 2 must be discovered by scanning")
    confidence: float = Field(ge=0.0, le=1.0, description="Honest confidence. Use below 0.7 when the task text is ambiguous about how much navigation it requires.")
    justification: str = Field(description="One sentence explaining the scores")


ANNOTATOR_INSTRUCTIONS = """\
You score WebArena web-agent tasks on four difficulty dimensions for a study of token budgets.
 
PAGES TO TRAVERSE - how many distinct pages must be visited?
    0 = everything needed is on the starting page
    1 = two or three pages
    2 = four or more, or an unbounded search across pages
 
RETRIEVAL TYPE - what must be done with the information?
    0 = read a single stated value
    1 = compare or filter a small number of items
    2 = aggregate, count, or reason over a set of items
 
INTERACTION - what must be done to the site?
    0 = read-only; nothing on the site changes
    1 = one form submission or click sequence
    2 = a multi-step state change (create, edit, delete, configure)
 
TARGET LOCATABILITY - how hard is the target to find?
    0 = named explicitly in the task, e.g. "the Sprite Stasis Ball"
    1 = derivable from what is on the page
    2 = must be discovered by scanning or searching
 
You measure the task's STRUCTURAL DEMANDS, not an agent's capability. Judge on
the number of distinct interactions and the reasoning depth required, not on
sentence length or surface wording.
 
You may also be shown OBSERVED COST: how many steps and distinct pages a
baseline agent actually needed. Use it to calibrate PAGES TO TRAVERSE, which
is difficult to judge from text alone. Do NOT use it to infer the other three
dimensions, and note that an agent may wander - observed pages is an upper
bound on what the task requires, not the exact number.
 
Give an honest confidence. Use below 0.7 when the task text leaves the amount
of navigation genuinely ambiguous.
"""

def tier_of(total: int) -> str:
    if total <= 2:
        return "Easy"
    if total <= 5:
        return "Medium"
    return "Hard"
 

annotator = W.make_agent(ANNOTATOR_INSTRUCTIONS, TaskAnnotation, label="annotator")
 
DIMENSIONS = ["pages_to_traverse", "retrieval_type", "interaction", "target_locatability"]


#Template normalisation
 
def normalise_template(intent: str) -> str:
    t = intent
    t = re.sub(r'"[^"]*"', "X", t)
    t = re.sub(r"'[^']*'", "X", t)
    t = re.sub(r"\b[A-Z][a-z]+_[A-Z][a-z]+\d*\b", "USER", t)
    t = re.sub(r"\b[a-zA-Z]+\d+\b", "USER", t)
    t = re.sub(r"\$?\d[\d,.]*\b", "N", t)
    t = re.sub(r"\bsubreddit \w+", "subreddit X", t)
    t = re.sub(r"\bforum \w+", "forum X", t)
    t = re.sub(r"\br/\w+", "r/X", t)
    return re.sub(r"\s+", " ", t).strip().lower()



#Observed cost from a collection file
 
def load_observed_cost(episodes_path: str | None) -> dict:
    if not episodes_path or not Path(episodes_path).exists():
        return {}
 
    by_task = defaultdict(list)
    with open(episodes_path) as f:
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
 
    observed = {}
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
 
        observed[tid] = {
            "median_steps": statistics.median(e["steps"] for e in acting),
            "max_steps": max(e["steps"] for e in acting),
            "median_tokens": statistics.median(e["total_tokens"] for e in acting),
            "distinct_urls": len(urls),
            "episodes": len(acting),
        }
    return observed

def format_observed(obs) -> str:
    if not obs:
        return ""
    return (
        f"\n Observed cost (a baseline single agent, across budget levels):\n"
        f"median steps taken: {obs['median_steps']:.0f}\n"
        f"maximum steps taken: {obs['max_steps']}\n"
        f"distinct pages visited: {obs['distinct_urls']}\n"
        f"median tokens: {obs['median_tokens']:.0f}\n"
        f"Use this to calibrate PAGES TO TRAVERSE only. The agent may have wandered, so distinct pages is an upper bound."
    )


#Annotation
 
def annotate(cfg: dict, category: str, observed) -> dict:
    intent = cfg.get("intent", "")
    eval_criteria = json.dumps(cfg.get("eval", {}))[:1200]
    sites = cfg.get("sites", [])
    site = sites[0] if sites else "unknown"
 
    prompt = (f"Site: {site}\n"
              f"Intent: {intent}\n"
              f"Evaluation criteria: {eval_criteria}"
              f"{format_observed(observed)}")
 
    a = W.call_agent(annotator, prompt).output
    total = sum(getattr(a, d) for d in DIMENSIONS)
 
    row = {
        "task_id": cfg.get("task_id"),
        "site": site,
        "intent": intent,
        "template": normalise_template(intent),
        "pages_to_traverse": a.pages_to_traverse,
        "retrieval_type": a.retrieval_type,
        "interaction": a.interaction,
        "target_locatability": a.target_locatability,
        "rubric_total": total,
        "difficulty_tier": tier_of(total),
        "task_category": category,
        "confidence": a.confidence,
        "justification": a.justification,
    }
    if observed:
        row["observed"] = observed
    return row


#Post-processing
 
def harmonise_templates(rows: list) -> int:
    groups = defaultdict(list)
    for r in rows:
        groups[r["template"]].append(r)
 
    adjusted = 0
    for tmpl, grp in groups.items():
        if len(grp) < 2:
            continue
        medians = {d: int(statistics.median(r[d] for r in grp))
                   for d in DIMENSIONS}
        for r in grp:
            changed = [d for d in DIMENSIONS if r[d] != medians[d]]
            if not changed:
                r["template_adjusted"] = False
                continue
            for d in changed:
                r[d] = medians[d]
            r["rubric_total"] = sum(r[d] for d in DIMENSIONS)
            r["difficulty_tier"] = tier_of(r["rubric_total"])
            r["template_adjusted"] = True
            r["adjusted_dimensions"] = changed
            adjusted += 1
    return adjusted


def flag_for_review(rows: list, threshold: float) -> None:
    for r in rows:
        reasons = []
 
        if r["confidence"] < threshold:
            reasons.append(f"low confidence ({r['confidence']:.2f})")
 
        if r.get("template_adjusted"):
            reasons.append(
                f"disagreed with template median on "
                f"{', '.join(r.get('adjusted_dimensions', []))}")
 
        #Score contradicts observed navigation
        obs = r.get("observed")
        if obs:
            pages = r["pages_to_traverse"]
            urls = obs["distinct_urls"]
            if pages == 0 and urls >= 3:
                reasons.append(f"scored 0 pages but agent visited {urls}")
            elif pages == 2 and urls <= 1:
                reasons.append(f"scored 2 pages but agent visited {urls}")
 
        r["needs_review"] = bool(reasons)
        r["review_reasons"] = reasons


#Reporting
 
def report(rows: list, sites: list, threshold: float) -> None:
    n = len(rows)
    if not n:
        print("No rows annotated.")
        return
 
    print(f"\n{'=' * 70}\nTier distribution\n{'=' * 70}")
    tiers = Counter(r["difficulty_tier"] for r in rows)
    for t in ("Easy", "Medium", "Hard"):
        c = tiers.get(t, 0)
        bar = "#" * int(40 * c / n)
        print(f"{t:8s} {c:4d}  ({c/n:>4.0%})  {bar}")

 
    print(f"\n{'=' * 70}\nTier by site\n{'=' * 70}")
    for site in sites:
        rs = [r for r in rows if r["site"] == site]
        if not rs:
            continue
        c = Counter(r["difficulty_tier"] for r in rs)
        print(f"   {site:16s} E:{c.get('Easy',0):>3}  "
              f"M:{c.get('Medium',0):>3}  H:{c.get('Hard',0):>3}   "
              f"(n={len(rs)})")
 
    print(f"\n{'=' * 70}\nCategory distribution\n{'=' * 70}")
    for cat, c in Counter(r["task_category"] for r in rows).most_common():
        print(f"   {cat:25s} {c:>4}")
 
    print(f"\n{'=' * 70}\nDimension scores\n{'=' * 70}")
    for d in DIMENSIONS:
        dist = Counter(r[d] for r in rows)
        print(f"   {d:22s} 0:{dist.get(0,0):>4}  1:{dist.get(1,0):>4}  "
              f"2:{dist.get(2,0):>4}")
 
    # Observed-cost correlation
    with_obs = [r for r in rows if r.get("observed")]
    if len(with_obs) >= 10:
        print(f"\n{'=' * 70}\nTier vs Observed cost (validity check)\n{'=' * 70}")
        for t in ("Easy", "Medium", "Hard"):
            sub = [r for r in with_obs if r["difficulty_tier"] == t]
            if not sub:
                continue
            print(f"   {t:8s} n={len(sub):>3}  "
                  f"median steps {statistics.median(r['observed']['median_steps'] for r in sub):>5.1f}  "
                  f"median tokens {statistics.median(r['observed']['median_tokens'] for r in sub):>8.0f}  "
                  f"median pages {statistics.median(r['observed']['distinct_urls'] for r in sub):>4.1f}")
 
    n_review = sum(1 for r in rows if r.get("needs_review"))
    n_adjusted = sum(1 for r in rows if r.get("template_adjusted"))
    print(f"\n{'=' * 70}\nReview queue\n{'=' * 70}")
    print(f"harmonised to template median: {n_adjusted}")
    print(f"flagged for review: {n_review}  "
          f"({n_review/n:.0%})")
    if n_review:
        print(f"\nReasons:")
        all_reasons = Counter()
        for r in rows:
            for reason in r.get("review_reasons", []):
                key = reason.split("(")[0].strip()
                all_reasons[key] += 1
        for reason, c in all_reasons.most_common():
            print(f" {reason:45s} {c:>4}")
        print(f"\n First 15 flagged tasks:")
        for r in [x for x in rows if x.get("needs_review")][:15]:
            print(f"{r['task_id']:>4} {r['difficulty_tier']:6s} "
                  f"{r['intent'][:44]}")
            print(f"{'; '.join(r['review_reasons'])[:66]}")



#Modes
 
def do_reband(path: str) -> None:
    rows = [json.loads(l) for l in open(path) if l.strip()]
    changed = 0
    for r in rows:
        r["rubric_total"] = sum(r[d] for d in DIMENSIONS)
        new_tier = tier_of(r["rubric_total"])
        if new_tier != r.get("difficulty_tier"):
            changed += 1
        r["difficulty_tier"] = new_tier
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"Re-banded {len(rows)} tasks, {changed} tiers changed")
    tiers = Counter(r["difficulty_tier"] for r in rows)
    for t in ("Easy", "Medium", "Hard"):
        c = tiers.get(t, 0)
        print(f"{t:8s} {c:>4}  ({c/len(rows):.0%})")
 
 
def do_gold_set(n: int, out: str, seed: int) -> None:
    configs = W.load_configs()
    pools = W.single_site_tasks()
    rng = random.Random(seed)
 
    per_site = max(1, n // len(pools))
    chosen = []
    for site, ids in pools.items():
        chosen.extend(rng.sample(ids, min(per_site, len(ids))))
 
    by_id = {c["task_id"]: c for c in configs if "task_id" in c}
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
 
    with out_path.open("w") as f:
        for tid in sorted(chosen):
            cfg = by_id.get(tid)
            if not cfg:
                continue
            f.write(json.dumps({
                "task_id": tid,
                "site": cfg["sites"][0],
                "intent": cfg.get("intent", ""),
                "pages_to_traverse": None,
                "retrieval_type": None,
                "interaction": None,
                "target_locatability": None,
            }) + "\n")
 
    print(f"Wrote {len(chosen)} tasks to {out_path}")


# Main
 
def _sample(n: int, sites: list, seed: int) -> list:
    pools = W.single_site_tasks(sites)
    rng = random.Random(seed)
    ids = []
    for site in sites:
        pool = pools.get(site, [])
        if pool:
            ids.extend(rng.sample(pool, min(n, len(pool))))
    return sorted(ids)
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", type=int)
    ap.add_argument("--n", type=int, default=999, help="tasks per site")
    ap.add_argument("--sites", nargs="+", default=W.SITES)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--intents", default=None, help="task_intents.json from extract_task_intents.py")
    ap.add_argument("--episodes", default=None, help="e.g. ../data/raw/strategy_a.jsonl")
    ap.add_argument("--review-threshold", type=float, default=0.75)
    ap.add_argument("--no-harmonise", action="store_true", help="skip template harmonisation (not recommended)")
    ap.add_argument("--out", default="../data/tasks/task_metadata.jsonl")
    ap.add_argument("--reband", default=None, help="re-derive tiers in an existing file and exit")
    ap.add_argument("--gold-set", type=int, default=None, help="write N tasks for hand-labelling and exit")
    args = ap.parse_args()
 
    if args.reband:
        do_reband(args.reband)
        return
    if args.gold_set:
        do_gold_set(args.gold_set, args.out, args.seed)
        return
 
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
 
    #Categories from extract_task_intents.py
    categories = {}
    if args.intents and Path(args.intents).exists():
        data = json.load(open(args.intents))
        for site_data in data.get("sites", {}).values():
            for cat, info in site_data.get("categories", {}).items():
                for t in info.get("tasks", []):
                    categories[t["task_id"]] = cat
        print(f"[annotate] categories loaded for {len(categories)} tasks")
 
    observed = load_observed_cost(args.episodes)
    if observed:
        print(f"[annotate] observed cost loaded for {len(observed)} tasks from {args.episodes}")
 
    task_ids = args.tasks or _sample(args.n, args.sites, args.seed)
    by_id = {c["task_id"]: c for c in W.load_configs() if "task_id" in c}
 
    print(f"[annotate] tasks: {len(task_ids)}")
    print(f"[annotate] output: {out_path}\n")
 
    rows = []
    for i, tid in enumerate(task_ids, 1):
        cfg = by_id.get(tid)
        if cfg is None:
            print(f"[annotate] {i}/{len(task_ids)} task {tid}: NOT FOUND")
            continue
        try:
            row = annotate(cfg, categories.get(tid, "unknown"),
                           observed.get(tid))
        except Exception as e:
            print(f"[annotate] {i}/{len(task_ids)} task {tid}: "
                  f"{type(e).__name__}: {e}")
            continue
 
        rows.append(row)
        obs_mark = "*" if row.get("observed") else " "
        print(f"[annotate] {i}/{len(task_ids)} task {tid:>4} "
              f"{row['site']:15s} {row['rubric_total']}/8 "
              f"{row['difficulty_tier']:6s} {obs_mark} {row['task_category']}")
 
    #Post-process
    if not args.no_harmonise:
        adjusted = harmonise_templates(rows)
        print(f"\n[annotate] harmonised {adjusted} rows to their template median")
 
    flag_for_review(rows, args.review_threshold)
 
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
 
    report(rows, args.sites, args.review_threshold)
    print(f"\n[annotate] wrote {len(rows)} rows to {out_path}")
 
 
if __name__ == "__main__":
    main()