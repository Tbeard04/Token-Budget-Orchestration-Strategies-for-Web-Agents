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


def from_csv(gold_path: str, csv_path: str, out_path: str | None) -> None:
    rows = load_gold(gold_path)
    by_id = {r.get("task_id"): r for r in rows}

    #utf-8-sig tolerates the BOM Excel writes
    with open(csv_path, newline="", encoding="utf-8-sig") as f:
        csv_rows = list(csv.DictReader(f))

    missing = [c for c in ("task_id", *DIMENSIONS) if c not in (csv_rows[0] if csv_rows else {})]
    if missing:
        raise SystemExit(f"CSV is missing required columns: {missing}")

    filled, blank, bad, unknown = 0, [], [], []
    for cr in csv_rows:
        try:
            tid = int(str(cr["task_id"]).strip())
        except (ValueError, KeyError):
            continue
        target = by_id.get(tid)
        if target is None:
            unknown.append(tid)
            continue

        scores = {}
        for d in DIMENSIONS:
            v = (cr.get(d) or "").strip()
            if v == "":
                scores = {}
                blank.append(tid)
                break
            try:
                iv = int(float(v))
            except ValueError:
                scores = {}
                bad.append((tid, d, v))
                break
            if iv not in (0, 1, 2):
                scores = {}
                bad.append((tid, d, v))
                break
            scores[d] = iv

        if scores:
            target.update(scores)
            filled += 1

    dest = Path(out_path or gold_path)
    if dest.exists() and not out_path:
        backup = dest.with_suffix(dest.suffix + ".bak")
        shutil.copy2(dest, backup)
        print(f"Backed up {dest.name} -> {backup.name}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(rows)} rows to {dest}  ({filled} scored)")
    if blank:
        print(f"   {len(blank)} rows still blank: {blank}")
    if bad:
        print(f"   {len(bad)} rows with a value outside 0-2: {bad[:10]}")
    if unknown:
        print(f"   {len(unknown)} csv task_ids not in the gold set: {unknown[:10]}")

    check(str(dest))


# Checking
# AI-Generated
def check(gold_path: str) -> None:
    rows = load_gold(gold_path)
    done = [r for r in rows if all(r.get(d) is not None for d in DIMENSIONS)]
    todo = [r["task_id"] for r in rows if r not in done]

    print(f"\n{'=' * 66}\nGold set: {len(done)}/{len(rows)} labelled\n{'=' * 66}")
    if todo:
        print(f"still blank: {todo}")
    if not done:
        return

    for d in DIMENSIONS:
        dist = Counter(r[d] for r in done)
        print(f"   {d:22s} 0:{dist.get(0, 0):>3}  1:{dist.get(1, 0):>3}  "
              f"2:{dist.get(2, 0):>3}")

    totals = [sum(r[d] for d in DIMENSIONS) for r in done]
    tiers = Counter(tier_of(t) for t in totals)
    print(f"\n rubric total: mean {sum(totals) / len(totals):.2f}  "
          f"range {min(totals)}-{max(totals)}")
    for t in TIERS:
        c = tiers.get(t, 0)
        print(f"   {t:8s} {c:>3}  ({c / len(done):.0%})")

    # near-identical tasks must agree, or the gold set contradicts itself
    group_of = cluster_similar(done)
    groups = defaultdict(list)
    for r in done:
        gid = group_of.get(r["task_id"])
        if gid:
            groups[gid].append(r)

    clashes = []
    for gid, grp in groups.items():
        for d in DIMENSIONS:
            if len({r[d] for r in grp}) > 1:
                clashes.append((gid, d, {r["task_id"]: r[d] for r in grp}))

    if clashes:
        print(f"\n {len(clashes)} inconsistencies across near-identical tasks:")
        for gid, d, vals in clashes[:12]:
            print(f" {gid} differs on {d}: {vals}")
    elif groups:
        print(f"\n {len(groups)} near-identical groups: all consistent")
#AI-Generated

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True, help="gold set jsonl")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default=None,
                    help="where --from-csv writes; defaults to --gold in place")
    ap.add_argument("--to-csv", action="store_true")
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.to_csv:
        if not args.csv:
            ap.error("--to-csv needs --csv")
        to_csv(args.gold, args.csv)
    elif args.from_csv:
        if not args.csv:
            ap.error("--from-csv needs --csv")
        from_csv(args.gold, args.csv, args.out)
    elif args.check:
        check(args.gold)
    else:
        ap.error("pick one of --to-csv, --from-csv, --check")

if __name__ == "__main__":
    main()