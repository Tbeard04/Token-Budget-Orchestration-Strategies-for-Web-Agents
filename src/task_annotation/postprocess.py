"""
postprocess.py - consistency enforcement and the review queue
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from rubric import DIMENSIONS, tier_of, total_of


def harmonise_templates(rows: list, protect: set | None = None) -> int:
    #Force instances of one template to share the group median scores
    protect = protect or set()
    groups = defaultdict(list)
    for r in rows:
        groups[r["template"]].append(r)

    adjusted = 0
    for grp in groups.values():
        if len(grp) < 2:
            continue
        # human-labelled rows set the standard
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
            r["rubric_total"] = total_of(r)
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

        # a page score that contradicts observed navigation
        obs = r.get("observed")
        if obs:
            pages, urls = r["pages_to_traverse"], obs["distinct_urls"]
            if pages == 0 and urls >= 3:
                reasons.append(f"scored 0 pages but agent visited {urls}")
            elif pages == 2 and urls <= 1:
                reasons.append(f"scored 2 pages but agent visited {urls}")

        r["needs_review"] = bool(reasons)
        r["review_reasons"] = reasons


def reband(rows: list) -> int:
    #Re-derive totals and tiers from the stored dimension scores
    changed = 0
    for r in rows:
        r["rubric_total"] = total_of(r)
        new_tier = tier_of(r["rubric_total"])
        if new_tier != r.get("difficulty_tier"):
            changed += 1
        r["difficulty_tier"] = new_tier
    return changed