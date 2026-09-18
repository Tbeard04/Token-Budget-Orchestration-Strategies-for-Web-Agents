"""
annotate_tasks.py - derive a difficulty tier for each WebArena task.

WebArena ships no difficulty labels. RQ2 asks how the three strategies differ
in sensitivity to task difficulty, so those labels must be derived here. Two
properties make them defensible:

1. Scored from the TASK TEXT (plus observed navigation cost), never from
   episode SUCCESS. If difficulty were defined by "Strategy A failed", then
   "harder tasks have lower success" would be true by construction.

2. Scored on four CONCRETE dimensions rather than a holistic judgement, and
   calibrated against a human-labelled gold set that is injected into the
   annotator as few-shot exemplars.

THE RUBRIC (each dimension 0-2, total 0-8)
  PAGES TO TRAVERSE    0 single page   1 two-three pages   2 four+/unbounded
  RETRIEVAL TYPE       0 one value     1 compare/filter     2 aggregate/count
  INTERACTION          0 read-only     1 one form/click     2 multi-step change
  TARGET LOCATABILITY  0 named         1 derivable          2 must be discovered

  0-2 Easy   3-5 Medium   6-8 Hard


HOW THE GOLD SET IS USED
------------------------
Of the 50 hand-labelled tasks, only --n-exemplars (default 20) are shown to
the annotator. The remaining 30 are HELD OUT: the annotator never sees their
labels, so Cohen's kappa on the held-out subset measures genuine agreement
rather than the model's ability to echo examples it was shown. Every output
row carries gold_exemplar / gold_heldout so cohen_kappa.py can split them.


RUN ORDER (from src/task_analysis/)
-----------------------------------
 1. Write the blank gold set, then fill in the four scores by hand:
      python annotate_tasks.py --gold-set 50 \
        --episodes ../../data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_A/strategy_a.jsonl \
        --gold-out ../../data/raw/annotation_tests/run2/gold_set_50.jsonl

 2. Annotate all tasks, anchored on the hand labels:
      python annotate_tasks.py --n 999 \
        --intents ../../data/tasks/task_intents.json \
        --episodes ../../data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_A/strategy_a.jsonl \
        --gold-labels ../../data/raw/annotation_tests/run2/gold_set_50.jsonl \
        --out ../../data/raw/annotation_tests/run2/tasks_annotated_run2.jsonl

 3. Agreement:
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
import random
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wa_env as W


# ---------------------------------------------------------------------------
# Rubric output
# The LLM scores four dimensions and nothing else. Category comes from
# extract_task_intents.py, which is regex-based, self-tested and consistent.
# ---------------------------------------------------------------------------

class TaskAnnotation(BaseModel):
    pages_to_traverse: int = Field(ge=0, le=2, description="0 single page, 1 two-to-three pages, 2 four-plus or unbounded")
    retrieval_type: int = Field(ge=0, le=2, description="0 read one value, 1 compare or filter a few, 2 aggregate or count over a set")
    interaction: int = Field(ge=0, le=2, description="0 read-only, 1 one form or click sequence, 2 multi-step state change")
    target_locatability: int = Field(ge=0, le=2, description="0 target named explicitly, 1 derivable from the page, 2 must be discovered by scanning")
    confidence: float = Field(ge=0.0, le=1.0, description="Honest confidence. Use below 0.7 when the task text is ambiguous about how much navigation it requires.")


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

DIMENSIONS = ["pages_to_traverse", "retrieval_type", "interaction", "target_locatability"]


def tier_of(total: int) -> str:
    if total <= 2:
        return "Easy"
    if total <= 5:
        return "Medium"
    return "Hard"


# ---------------------------------------------------------------------------
# Gold-set exemplars
# Only a subset of the hand-labelled tasks becomes exemplars; the rest are
# held out so agreement can be measured on labels the model never saw.
# ---------------------------------------------------------------------------

def load_gold_labels(path: str | None) -> dict:
    """Read hand-labelled rows. Rows with any unscored dimension are skipped."""
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[annotate] gold labels not found: {p}")
        return {}

    gold, blank = {}, 0
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        if any(r.get(d) is None for d in DIMENSIONS):
            blank += 1
            continue
        try:
            scores = {d: int(r[d]) for d in DIMENSIONS}
        except (TypeError, ValueError):
            blank += 1
            continue
        if any(not 0 <= v <= 2 for v in scores.values()):
            print(f"[annotate] task {r.get('task_id')}: score outside 0-2, skipped")
            continue
        scores["task_id"] = r.get("task_id")
        scores["site"] = r.get("site", "")
        scores["intent"] = r.get("intent", "")
        gold[r.get("task_id")] = scores

    print(f"[annotate] gold labels: {len(gold)} complete"
          + (f", {blank} still blank (skipped)" if blank else ""))
    return gold


def split_exemplars(gold: dict, n_exemplars: int, seed: int) -> tuple[dict, dict]:
    """Stratify the gold set by site, then take n_exemplars for the prompt."""
    if not gold:
        return {}, {}
    if n_exemplars >= len(gold):
        print(f"[annotate] WARNING: all {len(gold)} gold tasks used as exemplars; "
              f"kappa will not have a held-out subset")
        return dict(gold), {}

    by_site = defaultdict(list)
    for tid in sorted(gold):
        by_site[gold[tid]["site"]].append(tid)

    rng = random.Random(seed)
    picked: list[int] = []
    sites = sorted(by_site)
    for site in sites:
        rng.shuffle(by_site[site])

    # round-robin across sites so exemplars stay site-balanced
    i = 0
    while len(picked) < n_exemplars:
        pool = by_site[sites[i % len(sites)]]
        if pool:
            picked.append(pool.pop())
        elif all(not by_site[s] for s in sites):
            break
        i += 1

    ex = {t: gold[t] for t in picked}
    held = {t: gold[t] for t in gold if t not in ex}
    return ex, held


def format_exemplars(ex: dict) -> str:
    if not ex:
        return ""
    rows = sorted(ex.values(), key=lambda r: (sum(r[d] for d in DIMENSIONS), r["site"]))
    lines = [
        "",
        "CALIBRATION EXAMPLES",
        "The tasks below were scored by the human researcher. They define the",
        "standard you must match. Apply the same level of strictness; do not",
        "score systematically higher or lower than these.",
        "",
    ]
    for r in rows:
        total = sum(r[d] for d in DIMENSIONS)
        lines.append(f"[{r['site']}] {r['intent'][:150]}")
        lines.append(
            f"   pages={r['pages_to_traverse']} retrieval={r['retrieval_type']} "
            f"interaction={r['interaction']} locatability={r['target_locatability']} "
            f"-> {total}/8 {tier_of(total)}"
        )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Template normalisation
# WebArena tasks are template-generated; instances of one template must not
# receive different scores.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Observed cost from one or more collection files
# Strategy A is listed first: it makes one call per step, so its step counts
# track task demand directly. Strategy B's are confounded by the pipeline
# dying earlier at the same budget, so it only fills gaps A does not cover.
# ---------------------------------------------------------------------------

def load_observed_cost(episode_paths) -> dict:
    if not episode_paths:
        return {}
    if isinstance(episode_paths, str):
        episode_paths = [episode_paths]

    observed: dict = {}
    for path in episode_paths:
        if not Path(path).exists():
            print(f"[annotate] episodes not found, skipped: {path}")
            continue

        by_task = defaultdict(list)
        with open(path) as f:
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

        added = 0
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

            stat = {
                "median_steps": statistics.median(e["steps"] for e in acting),
                "max_steps": max(e["steps"] for e in acting),
                "median_tokens": statistics.median(e.get("total_tokens", 0) for e in acting),
                "distinct_urls": len(urls),
                "episodes": len(acting),
                "source": Path(path).stem,
            }
            # first file wins: A's numbers are not overwritten by B's
            if observed.setdefault(tid, stat) is stat:
                added += 1

        print(f"[annotate] observed cost: +{added} tasks from {Path(path).name}")

    if observed:
        print(f"[annotate] observed cost loaded for {len(observed)} tasks total")
    return observed


def format_observed(obs) -> str:
    if not obs:
        return ""
    return (
        f"\nObserved cost (a baseline single agent, across budget levels):\n"
        f"median steps taken: {obs['median_steps']:.0f}\n"
        f"maximum steps taken: {obs['max_steps']}\n"
        f"distinct pages visited: {obs['distinct_urls']}\n"
        f"median tokens: {obs['median_tokens']:.0f}\n"
        f"Use this to calibrate PAGES TO TRAVERSE only. The agent may have "
        f"wandered, so distinct pages is an upper bound."
    )


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

def annotate(agent, cfg: dict, category: str, observed) -> dict:
    intent = cfg.get("intent", "")
    eval_criteria = json.dumps(cfg.get("eval", {}))[:1200]
    sites = cfg.get("sites", [])
    site = sites[0] if sites else "unknown"

    prompt = (f"Site: {site}\n"
              f"Intent: {intent}\n"
              f"Evaluation criteria: {eval_criteria}"
              f"{format_observed(observed)}")

    a = W.call_agent(agent, prompt).output
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
    }
    if observed:
        row["observed"] = observed
    return row


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def harmonise_templates(rows: list, protect: set | None = None) -> int:
    """Force instances of one template to share the group's median scores.

    Rows whose task_id is in `protect` (the hand-labelled exemplars) keep
    their scores and are excluded from the median, so a human label is never
    silently overwritten by the model's majority.
    """
    protect = protect or set()
    groups = defaultdict(list)
    for r in rows:
        groups[r["template"]].append(r)

    adjusted = 0
    for grp in groups.values():
        if len(grp) < 2:
            continue
        basis = [r for r in grp if r["task_id"] not in protect] or grp
        medians = {d: int(statistics.median(r[d] for r in basis)) for d in DIMENSIONS}
        for r in grp:
            if r["task_id"] in protect:
                r["template_adjusted"] = False
                continue
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
            reasons.append("disagreed with template median on "
                           + ", ".join(r.get("adjusted_dimensions", [])))

        obs = r.get("observed")
        if obs:
            pages, urls = r["pages_to_traverse"], obs["distinct_urls"]
            if pages == 0 and urls >= 3:
                reasons.append(f"scored 0 pages but agent visited {urls}")
            elif pages == 2 and urls <= 1:
                reasons.append(f"scored 2 pages but agent visited {urls}")

        r["needs_review"] = bool(reasons)
        r["review_reasons"] = reasons


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def report(rows: list, sites: list, threshold: float) -> None:
    n = len(rows)
    if not n:
        print("No rows annotated.")
        return

    print(f"\n{'=' * 70}\nTier distribution\n{'=' * 70}")
    tiers = Counter(r["difficulty_tier"] for r in rows)
    for t in ("Easy", "Medium", "Hard"):
        c = tiers.get(t, 0)
        print(f"{t:8s} {c:4d}  ({c/n:>4.0%})  {'#' * int(40 * c / n)}")

    print(f"\n{'=' * 70}\nTier by site\n{'=' * 70}")
    for site in sites:
        rs = [r for r in rows if r["site"] == site]
        if not rs:
            continue
        c = Counter(r["difficulty_tier"] for r in rs)
        print(f"   {site:16s} E:{c.get('Easy',0):>3}  M:{c.get('Medium',0):>3}  "
              f"H:{c.get('Hard',0):>3}   (n={len(rs)})")

    print(f"\n{'=' * 70}\nCategory distribution\n{'=' * 70}")
    for cat, c in Counter(r["task_category"] for r in rows).most_common():
        print(f"   {cat:25s} {c:>4}")

    print(f"\n{'=' * 70}\nDimension scores\n{'=' * 70}")
    for d in DIMENSIONS:
        dist = Counter(r[d] for r in rows)
        print(f"   {d:22s} 0:{dist.get(0,0):>4}  1:{dist.get(1,0):>4}  2:{dist.get(2,0):>4}")

    with_obs = [r for r in rows if r.get("observed")]
    if len(with_obs) >= 10:
        print(f"\n{'=' * 70}\nTier vs observed cost (validity check)\n{'=' * 70}")
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
    print(f"flagged for review: {n_review}  ({n_review/n:.0%})")
    if n_review:
        all_reasons = Counter()
        for r in rows:
            for reason in r.get("review_reasons", []):
                all_reasons[reason.split("(")[0].strip()] += 1
        print("\nReasons:")
        for reason, c in all_reasons.most_common():
            print(f"   {reason:45s} {c:>4}")
        print("\nFirst 15 flagged tasks:")
        for r in [x for x in rows if x.get("needs_review")][:15]:
            print(f"   {r['task_id']:>4} {r['difficulty_tier']:6s} {r['intent'][:44]}")
            print(f"        {'; '.join(r['review_reasons'])[:66]}")


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

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


def _allocate(n: int, pools: dict) -> dict:
    """Largest-remainder allocation of n tasks proportional to pool size."""
    total = sum(len(v) for v in pools.values())
    if not total:
        return {}
    sites = sorted(pools, key=lambda s: -len(pools[s]))
    exact = {s: n * len(pools[s]) / total for s in sites}
    alloc = {s: min(int(exact[s]), len(pools[s])) for s in sites}
    left = n - sum(alloc.values())
    for s in sorted(sites, key=lambda s: -(exact[s] - int(exact[s]))):
        if left > 0 and alloc[s] < len(pools[s]):
            alloc[s] += 1
            left -= 1
    # any remainder (a site ran out) spills to whichever site still has room
    i = 0
    while left > 0 and i < 10_000:
        s = sites[i % len(sites)]
        if alloc[s] < len(pools[s]):
            alloc[s] += 1
            left -= 1
        i += 1
    return alloc


def do_gold_set(n: int, out: str, seed: int, sites: list, episode_paths) -> None:
    """Write exactly n blank rows, stratified by site, for hand-labelling."""
    pools = {s: ids for s, ids in W.single_site_tasks(sites).items() if ids}
    alloc = _allocate(n, pools)
    rng = random.Random(seed)

    chosen: list[int] = []
    for site in sorted(alloc):
        chosen.extend(rng.sample(pools[site], alloc[site]))

    observed = load_observed_cost(episode_paths)
    by_id = {c["task_id"]: c for c in W.load_configs() if "task_id" in c}

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with out_path.open("w") as f:
        for tid in sorted(chosen):
            cfg = by_id.get(tid)
            if not cfg:
                continue
            row = {
                "task_id": tid,
                "site": cfg["sites"][0],
                "intent": cfg.get("intent", ""),
                "eval_criteria": json.dumps(cfg.get("eval", {}))[:600],
                # fill these four in by hand, each 0, 1 or 2
                "pages_to_traverse": None,
                "retrieval_type": None,
                "interaction": None,
                "target_locatability": None,
            }
            obs = observed.get(tid)
            if obs:
                # shown so the human scores from the same evidence the
                # annotator gets; otherwise kappa compares two instruments
                row["observed_hint"] = {
                    "median_steps": obs["median_steps"],
                    "max_steps": obs["max_steps"],
                    "distinct_urls": obs["distinct_urls"],
                }
            f.write(json.dumps(row) + "\n")
            written += 1

    print(f"\nWrote {written} blank rows to {out_path}")
    for site in sorted(alloc):
        print(f"   {site:16s} {alloc[site]:>3}  (pool {len(pools[site])})")
    print(f"\nTask ids: {' '.join(str(t) for t in sorted(chosen))}")
    print("\nFill in pages_to_traverse, retrieval_type, interaction and")
    print("target_locatability on every row (0, 1 or 2). Leave nothing null:")
    print("blank rows are skipped, which shrinks the gold set.")
    print("\nRubric:")
    print("  pages_to_traverse    0 one page        1 two-three      2 four+/unbounded")
    print("  retrieval_type       0 one value       1 compare/filter 2 aggregate/count")
    print("  interaction          0 read-only       1 one form/click 2 multi-step change")
    print("  target_locatability  0 named in task   1 derivable      2 must be discovered")


def do_apply_gold(pred_path: str, gold: dict, out: str) -> None:
    """Overwrite the model's scores with the human ones for gold tasks.

    Run this AFTER cohen_kappa.py: it destroys the comparison by construction.
    """
    rows = [json.loads(l) for l in open(pred_path) if l.strip()]
    replaced = 0
    for r in rows:
        g = gold.get(r.get("task_id"))
        if not g:
            continue
        if any(r[d] != g[d] for d in DIMENSIONS):
            replaced += 1
        for d in DIMENSIONS:
            r[d] = g[d]
        r["rubric_total"] = sum(r[d] for d in DIMENSIONS)
        r["difficulty_tier"] = tier_of(r["rubric_total"])
        r["label_source"] = "human"
        r["needs_review"] = False
        r["review_reasons"] = []
    for r in rows:
        r.setdefault("label_source", "model")

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_human = sum(1 for r in rows if r["label_source"] == "human")
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"   human labels applied: {n_human} ({replaced} differed from the model)")
    tiers = Counter(r["difficulty_tier"] for r in rows)
    for t in ("Easy", "Medium", "Hard"):
        c = tiers.get(t, 0)
        print(f"   {t:8s} {c:>4}  ({c/max(len(rows),1):.0%})")


def _sample(n: int, sites: list, seed: int) -> list:
    pools = W.single_site_tasks(sites)
    rng = random.Random(seed)
    ids = []
    for site in sites:
        pool = pools.get(site, [])
        if pool:
            ids.extend(rng.sample(pool, min(n, len(pool))))
    return sorted(ids)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="*", type=int)
    ap.add_argument("--n", type=int, default=999, help="tasks per site")
    ap.add_argument("--sites", nargs="+", default=W.SITES)
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
        do_gold_set(args.gold_set, args.gold_out, args.seed,
                    args.sites, args.episodes)
        return

    if args.apply_gold:
        gold = load_gold_labels(args.gold_labels)
        if not gold:
            ap.error("--apply-gold needs a --gold-labels file with completed scores")
        do_apply_gold(args.apply_gold, gold, args.out)
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Calibration: some gold tasks become exemplars, the rest are held out
    gold = load_gold_labels(args.gold_labels)
    exemplars, heldout = split_exemplars(gold, args.n_exemplars, args.seed)
    if gold:
        print(f"[annotate] exemplars shown to annotator: {len(exemplars)}  "
              f"held out for kappa: {len(heldout)}")

    instructions = ANNOTATOR_INSTRUCTIONS + format_exemplars(exemplars)
    agent = W.make_agent(instructions, TaskAnnotation, label="annotator")

    # Categories from extract_task_intents.py
    categories = {}
    if args.intents and Path(args.intents).exists():
        data = json.load(open(args.intents))
        for site_data in data.get("sites", {}).values():
            for cat, info in site_data.get("categories", {}).items():
                for t in info.get("tasks", []):
                    categories[t["task_id"]] = cat
        print(f"[annotate] categories loaded for {len(categories)} tasks")

    observed = load_observed_cost(args.episodes)

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
            row = annotate(agent, cfg, categories.get(tid, "unknown"),
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
        # exemplar rows carry the human label's authority, so they anchor the
        # median rather than being pulled towards it
        adjusted = harmonise_templates(rows, protect=set(exemplars))
        print(f"\n[annotate] harmonised {adjusted} rows to their template median")

    flag_for_review(rows, args.review_threshold)

    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    report(rows, args.sites, args.review_threshold)
    print(f"\n[annotate] wrote {len(rows)} rows to {out_path}")
    if heldout:
        print(f"[annotate] next: cohen_kappa.py --gold {args.gold_labels} "
              f"--pred {out_path}")


if __name__ == "__main__":
    main()