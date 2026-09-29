"""
reporting.py - terminal summaries of an annotation run
"""
from __future__ import annotations

import statistics
from collections import Counter

from rubric import DIMENSIONS, TIERS

#function to print the rule
def _rule(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")

#function to print the tier distribution
def tier_distribution(rows: list) -> None:
    n = len(rows)
    _rule("Tier distribution")
    tiers = Counter(r["difficulty_tier"] for r in rows)
    for t in TIERS:
        c = tiers.get(t, 0)
        print(f"{t:8s} {c:4d}  ({c / n:>4.0%})  {'#' * int(40 * c / n)}")

#function to print the tier by site
def tier_by_site(rows: list, sites: list) -> None:
    _rule("Tier by site")
    for site in sites:
        rs = [r for r in rows if r["site"] == site]
        if not rs:
            continue
        c = Counter(r["difficulty_tier"] for r in rs)
        print(f" {site:16s} E:{c.get('Easy', 0):>3}  M:{c.get('Medium', 0):>3}  "
              f"H:{c.get('Hard', 0):>3}   (n={len(rs)})")

#function to print the category distribution
def category_distribution(rows: list) -> None:
    _rule("Category distribution")
    for cat, c in Counter(r["task_category"] for r in rows).most_common():
        print(f"{cat:25s} {c:>4}")

#function to print the dimension spread
def dimension_spread(rows: list) -> None:
    _rule("Dimension scores")
    for d in DIMENSIONS:
        dist = Counter(r[d] for r in rows)
        print(f" {d:22s} 0:{dist.get(0, 0):>4}  1:{dist.get(1, 0):>4}  "
              f"2:{dist.get(2, 0):>4}")

#function to print the validity check
def validity_check(rows: list) -> None:
    #Tier against observed cost. Expect steps and tokens to rise with tier
    with_obs = [r for r in rows if r.get("observed")]
    if len(with_obs) < 10:
        return
    _rule("Tier vs observed cost (validity check)")
    for t in TIERS:
        sub = [r for r in with_obs if r["difficulty_tier"] == t]
        if not sub:
            continue
        print(f" {t:8s} n={len(sub):>3}  "
              f"median steps {statistics.median(r['observed']['median_steps'] for r in sub):>5.1f}  "
              f"median tokens {statistics.median(r['observed']['median_tokens'] for r in sub):>8.0f}  "
              f"median pages {statistics.median(r['observed']['distinct_urls'] for r in sub):>4.1f}")

#function to print the review queue
def review_queue(rows: list, show: int = 15) -> None:
    n = len(rows)
    n_review = sum(1 for r in rows if r.get("needs_review"))
    n_adjusted = sum(1 for r in rows if r.get("template_adjusted"))

    #print the review queue
    _rule("Review queue")
    print(f"harmonised to template median: {n_adjusted}")
    print(f"flagged for review: {n_review} ({n_review / n:.0%})")
    if not n_review:
        return
    #create a counter to store the reasons
    all_reasons = Counter()
    for r in rows:
        for reason in r.get("review_reasons", []):
            #split the reason by the first parenthesis and strip the whitespace
            all_reasons[reason.split("(")[0].strip()] += 1
    print("\nReasons:")
    for reason, c in all_reasons.most_common():
        print(f" {reason:45s} {c:>4}")

    print(f"\nFirst {show} flagged tasks:")
    for r in [x for x in rows if x.get("needs_review")][:show]:
        print(f" {r['task_id']:>4} {r['difficulty_tier']:6s} {r['intent'][:44]}")
        print(f" {'; '.join(r['review_reasons'])[:66]}")


#function to print the report
def report(rows: list, sites: list) -> None:
    if not rows:
        print("No rows annotated.")
        return
    tier_distribution(rows)
    tier_by_site(rows, sites)
    category_distribution(rows)
    dimension_spread(rows)
    validity_check(rows)
    review_queue(rows)