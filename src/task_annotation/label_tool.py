"""
label_tool.py - hand-label the gold set in a spreadsheet instead of raw JSONL so its easier to work with.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from rubric import DIMENSIONS, RUBRIC_SUMMARY, TIERS, normalise_template, tier_of

COLUMNS = [
    "task_id", "site", "group", "intent", "eval_type", "eval_target",
    "obs_median_steps", "obs_max_steps", "obs_urls",
    "pages_to_traverse", "retrieval_type", "interaction", "target_locatability",
]


# Making the evaluation criteria readable
def summarise_eval(raw: str) -> tuple[str, str]:
    #Return (eval_type, target) from the stored criteria JSON.
    if not raw:
        return "", ""
    try:
        e = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'"eval_types":\s*\[([^\]]*)\]', raw)
        types = m.group(1).replace('"', "").replace(" ", "") if m else "?"
        return f"{types} (truncated)", raw[:120]

    types = "+".join(e.get("eval_types") or [])

    ans = e.get("reference_answers")
    if ans:
        for key in ("exact_match", "must_include", "fuzzy_match"):
            if key in ans:
                v = ans[key]
                v = ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)
                return types, f"{key}: {v[:200]}"

    if e.get("reference_url"):
        return types, f"url: {e['reference_url'][:200]}"

    ph = e.get("program_html") or []
    if ph:
        urls = {p.get("url", "") for p in ph}
        return types, f"DOM check x{len(ph)} on {len(urls)} page(s)"

    return types, ""

# JSONL <--> CSV conversion
def load_gold(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def cluster_similar(rows: list, threshold: float = 0.85) -> dict:
    #Single-link cluster task ids by intent-template similarity.
    tmpl = {r["task_id"]: normalise_template(r.get("intent", "")) for r in rows}
    ids = sorted(tmpl)
    parent = {i: i for i in ids}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a_i, a in enumerate(ids):
        for b in ids[a_i + 1:]:
            if SequenceMatcher(None, tmpl[a], tmpl[b]).ratio() >= threshold:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra

    clusters = defaultdict(list)
    for i in ids:
        clusters[find(i)].append(i)

    label, out = {}, {}
    for n, (root, members) in enumerate(sorted(clusters.items()), 1):
        if len(members) < 2:
            continue
        label[root] = f"G{len(label) + 1}"
        for m in members:
            out[m] = label[root]
    return out

def to_csv(gold_path: str, csv_path: str, threshold: float = 0.85) -> None:
    rows = load_gold(gold_path)
    group_of = cluster_similar(rows, threshold)

    ordered = sorted(rows, key=lambda r: (
        group_of.get(r.get("task_id"), "ZZ"),
        r.get("site", ""),
        r.get("task_id", 0),
    ))

    counts = Counter(normalise_template(r.get("intent", "")) for r in rows)

    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for r in ordered:
            etype, etarget = summarise_eval(r.get("eval_criteria", ""))
            obs = r.get("observed_hint") or {}
            gid = group_of.get(r.get("task_id"), "")
            w.writerow({
                "task_id": r.get("task_id"),
                "site": r.get("site", ""),
                "group": gid,
                "intent": r.get("intent", ""),
                "eval_type": etype,
                "eval_target": etarget,
                "obs_median_steps": obs.get("median_steps", ""),
                "obs_max_steps": obs.get("max_steps", ""),
                "obs_urls": obs.get("distinct_urls", ""),
                **{d: (r.get(d) if r.get(d) is not None else "") for d in DIMENSIONS},
            })

    # exact = sum(1 for c in counts.values() if c > 1)
    # n_groups = len(set(group_of.values()))
    print(f"Wrote {len(ordered)} rows to {out}")