"""
gold_set.py - human annotations for the difficulty
"""
from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import tasks
from observed_cost import load_observed_cost, summarise_hint
from rubric import DIMENSIONS, RUBRIC_SUMMARY, TIERS, tier_of, total_of


# Reading finished hand labels
def load_gold_labels(path: str | None) -> dict:
    #Read hand-labelled rows, keyed by task_id
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        print(f"[gold] labels not found: {p}")
        return {}

    gold, blank, bad = {}, [], []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue

        tid = r.get("task_id")
        if any(r.get(d) is None for d in DIMENSIONS):
            blank.append(tid)
            continue
        try:
            scores = {d: int(r[d]) for d in DIMENSIONS}
        except (TypeError, ValueError):
            bad.append(tid)
            continue
        if any(not 0 <= v <= 2 for v in scores.values()):
            bad.append(tid)
            continue

        scores["task_id"] = tid
        scores["site"] = r.get("site", "")
        scores["intent"] = r.get("intent", "")
        gold[tid] = scores

    print(f"[gold] {len(gold)} complete labels"
          + (f", {len(blank)} still blank: {blank[:20]}" if blank else "")
          + (f", {len(bad)} malformed: {bad[:20]}" if bad else ""))
    return gold


def split_exemplars(gold: dict, n_exemplars: int, seed: int) -> tuple[dict, dict]:
    #Take n_exemplars for the prompt, site-balanced; hold back the rest
    if not gold:
        return {}, {}
    if n_exemplars >= len(gold):
        print(f"[gold] WARNING: all {len(gold)} labels used as exemplars, so "
              f"kappa will have no held-out subset and will be inflated")
        return dict(gold), {}

    by_site = defaultdict(list)
    for tid in sorted(gold):
        by_site[gold[tid]["site"]].append(tid)

    rng = random.Random(seed)
    sites = sorted(by_site)
    for s in sites:
        rng.shuffle(by_site[s])

    picked: list = []
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
    #Render exemplars as a calibration block for the system prompt
    if not ex:
        return ""
    rows = sorted(ex.values(), key=lambda r: (total_of(r), r["site"]))
    lines = [
        "",
        "CALIBRATION EXAMPLES",
        "The tasks below were scored by the human researcher. They define the",
        "standard you must match. Apply the same level of strictness; do not",
        "score systematically higher or lower than these.",
        "",
    ]
    for r in rows:
        t = total_of(r)
        lines.append(f"[{r['site']}] {r['intent'][:150]}")
        lines.append(
            f"   pages={r['pages_to_traverse']} retrieval={r['retrieval_type']} "
            f"interaction={r['interaction']} locatability={r['target_locatability']} "
            f"-> {t}/8 {tier_of(t)}"
        )
    lines.append("")
    return "\n".join(lines)


# Writing the blank gold set
def write_gold_set(n: int, out: str, seed: int, sites: list, episode_paths) -> None:
    #Write exactly n blank rows, stratified by site, for hand-labelling
    site_pools = tasks.pools(sites)
    alloc = tasks.allocate(n, site_pools)
    rng = random.Random(seed)

    chosen: list = []
    for site in sorted(alloc):
        chosen.extend(rng.sample(site_pools[site], alloc[site]))

    observed = load_observed_cost(episode_paths)
    by_id = tasks.configs_by_id()

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
                "eval_criteria": json.dumps(cfg.get("eval", {}))[:1200],
                # fill these four in by hand, each 0, 1 or 2
                "pages_to_traverse": None,
                "retrieval_type": None,
                "interaction": None,
                "target_locatability": None,
            }
            obs = observed.get(tid)
            if obs:
                row["observed_hint"] = summarise_hint(obs)
            f.write(json.dumps(row) + "\n")
            written += 1

    print(f"\nWrote {written} blank rows to {out_path}")
    for site in sorted(alloc):
        print(f" {site:16s} {alloc[site]:>3}  (pool {len(site_pools[site])})")
    print(f"\nTask ids: {' '.join(str(t) for t in sorted(chosen))}")
    print(f"\nRubric:\n{RUBRIC_SUMMARY}")


# Merging hand labels into the final metadata
def apply_gold(pred_path: str, gold: dict, out: str) -> None:
    #Overwrite the model's scores with the human ones for gold tasks
    rows = [json.loads(l) for l in open(pred_path) if l.strip()]
    replaced = 0
    for r in rows:
        g = gold.get(r.get("task_id"))
        if not g:
            r.setdefault("label_source", "model")
            continue
        if any(r[d] != g[d] for d in DIMENSIONS):
            replaced += 1
        for d in DIMENSIONS:
            r[d] = g[d]
        r["rubric_total"] = total_of(r)
        r["difficulty_tier"] = tier_of(r["rubric_total"])
        r["label_source"] = "human"
        r["needs_review"] = False
        r["review_reasons"] = []

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    n_human = sum(1 for r in rows if r.get("label_source") == "human")
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"human labels applied: {n_human} ({replaced} differed from the model)")
    tiers = Counter(r["difficulty_tier"] for r in rows)
    for t in TIERS:
        c = tiers.get(t, 0)
        print(f" {t:8s} {c:>4}  ({c / max(len(rows), 1):.0%})")