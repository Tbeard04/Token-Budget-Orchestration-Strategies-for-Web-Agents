"""
This script is used to join the episodes with the metadata and risk levels.
"""
# Join both A, B and metadata to create single training dataset for C. 
# 4. Join the datasets on the task_id column
# 5. Save the joined dataset to a new file in the processed directory
# 6. each row needs to contain both A and B - IMPORTANT 
# split function to allow for different splits of the dataset (for analysis comparing difficulty tiers etc.. between A & B tasks) --> probably a seperate script for this


from __future__ import annotations
 
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
 
#Removed from every episode. Everything else is passed through
DROP_FIELDS = {"step_log"}
 
#attached from task_metadata.jsonl
#task_id is the join key
META_FIELDS = [
    "task_category",
    "pages_to_traverse",
    "retrieval_type",
    "interaction",
    "target_locatability",
    "rubric_total",
    "difficulty_tier",
    "label_source",
]
 
#Attached from task_risk_levels.jsonl
#risk_level is already on the episode
RISK_FIELDS = ["risk_reason"]
 
#Task-level fields that also exist on the episode
CROSS_CHECK = {"site": "site", "goal": "intent"}
 
#Budgets for the tasks
BUDGETS = [2000, 4000, 8000, 16000, 32000, 64000]
 
 
def load_jsonl(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


#the join function
def join(episodes: list[dict],
         meta: dict[int, dict],
         risk: dict[int, dict],
         keep_steps: bool = False) -> tuple[list[dict], dict]:
    #if keep_steps is True, the steps will be kept, otherwise they will be dropped
    drop = set() if keep_steps else DROP_FIELDS
    #report is a dictionary that will be used to store the missing metadata and risk, as well as the conflicts
    report = {
        "missing_meta": set(),
        "missing_risk": set(),
        "conflicts": defaultdict(list),
        "dropped_fields": sorted(drop),
    }
 
    out = [] #output list to store the joined episodes
    for e in episodes:
        tid = e["task_id"]
        m = meta.get(tid)
        r = risk.get(tid)
        if m is None:
            report["missing_meta"].add(tid)
        if r is None:
            report["missing_risk"].add(tid)
 
        row = {k: v for k, v in e.items() if k not in drop}
 
        #integrity: the episode and the metadata must agree about the task
        if m is not None:
            for ep_key, meta_key in CROSS_CHECK.items():
                a, b = e.get(ep_key), m.get(meta_key)
                if a is not None and b is not None and a != b:
                    report["conflicts"][ep_key].append(tid)
 
        if m is not None:
            for f in META_FIELDS:
                row[f] = m.get(f)
        else:
            for f in META_FIELDS:
                row[f] = None
 
        if r is not None:
            for f in RISK_FIELDS:
                row[f] = r.get(f)
        else:
            for f in RISK_FIELDS:
                row[f] = None
 
        out.append(row)
    #sort the missing metadata and risk, and the conflicts
    report["missing_meta"] = sorted(report["missing_meta"])
    report["missing_risk"] = sorted(report["missing_risk"])
    report["conflicts"] = {k: sorted(set(v)) for k, v in report["conflicts"].items()}
    return out, report


def _rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
 
 
def summarise(rows: list[dict], report: dict) -> None:
    strategies = sorted({r["strategy"] for r in rows})
 
    _rule("Join integrity")
    if report["missing_meta"]:
        print(f"   WARNING {len(report['missing_meta'])} task ids have no "
              f"difficulty label: {report['missing_meta'][:15]}")
    else:
        print("   every task id has a metadata row")
    #check if the difficulty tier is present
    no_tier = {r["task_id"] for r in rows if r.get("difficulty_tier") is None}
    if no_tier:
        print(f"   WARNING {len(no_tier)} task ids have a metadata row but no "
              f"difficulty_tier: {sorted(no_tier)[:15]}")
    else:
        print("   every task id has a difficulty tier")
    if report["missing_risk"]:
        print(f"   WARNING {len(report['missing_risk'])} task ids have no "
              f"risk reason: {report['missing_risk'][:15]}")
    else:
        print("   every task id has a risk classification")
    if report["conflicts"]:
        for field, ids in report["conflicts"].items():
            print(f"   WARNING {field} disagrees between episode and metadata "
                  f"for {len(ids)} tasks: {ids[:10]}")
    else:
        print("   episode and metadata agree on site and intent")
    if report["dropped_fields"]:
        print(f"   dropped per episode: {', '.join(report['dropped_fields'])}")
    
    #print the final table
    _rule("Final table")
    print(f"{'strategy':10s}{'episodes':>10}{'tasks':>8}{'successes':>11}"
          f"{'reward-eligible':>17}")
    for s in strategies:
        eps = [r for r in rows if r["strategy"] == s]
        clean = [r for r in eps if r["use_for_reward"]]
        print(f"{s:10s}{len(eps):>10}{len({r['task_id'] for r in eps}):>8}"
              f"{sum(1 for r in eps if r['success']):>11}"
              f"{len(clean):>17}")
    print(f"{'ALL':10s}{len(rows):>10}{len({r['task_id'] for r in rows}):>8}"
          f"{sum(1 for r in rows if r['success']):>11}"
          f"{sum(1 for r in rows if r['use_for_reward']):>17}")
 
    # The number that decides which basis the headline figures use. If the
    # exclusion rate is flat across budgets, the decontaminated set is safe as
    # the primary basis. If it climbs with budget, the rule censors the top of
    # the cost-performance curve and that needs stating in the methods.
    _rule("Exclusions by budget  (flat = decontaminated set is safe to headline)")
    print(f"{'budget':>8}{'episodes':>10}{'excluded':>10}{'excl %':>9}"
          f"{'successes':>11}{'of which excluded':>19}")
    rates = []
    for b in BUDGETS:
        eps = [r for r in rows if r["budget_level"] == b]
        if not eps:
            continue
        exc = [r for r in eps if not r["use_for_reward"]]
        suc = [r for r in eps if r["success"]]
        sx = [r for r in suc if not r["use_for_reward"]]
        rate = 100 * len(exc) / len(eps)
        rates.append(rate)
        print(f"{b:>8}{len(eps):>10}{len(exc):>10}{rate:>8.1f}%"
              f"{len(suc):>11}{len(sx):>19}")
    if rates:
        spread = max(rates) - min(rates)
        verdict = ("flat - no budget is preferentially censored"
                   if spread < 3 else
                   "SKEWED - the rule censors some budgets more than others; "
                   "state this in the methods")
        print(f"\n   spread {spread:.1f} percentage points: {verdict}")
 
    _rule("Difficulty tier x reward eligibility")
    tiers = ["Easy", "Medium", "Hard", None]
    print(f"{'tier':10s}" + "".join(f"{s:>12}" for s in strategies)
          + f"{'total':>10}")
    for t in tiers:
        eps = [r for r in rows if r["difficulty_tier"] == t and r["use_for_reward"]]
        if not eps and t is None:
            continue
        cells = "".join(
            f"{sum(1 for r in eps if r['strategy'] == s):>12}" for s in strategies)
        print(f"{str(t):10s}{cells}{len(eps):>10}")
 
    # Successes per tier is the cell count that actually limits the analysis:
    # a tier with a handful of successes cannot support a per-tier comparison.
    _rule("Reward-eligible SUCCESSES per tier  (small cells limit the analysis)")
    print(f"{'tier':10s}" + "".join(f"{s:>12}" for s in strategies))
    for t in ["Easy", "Medium", "Hard"]:
        cells = ""
        for s in strategies:
            n = sum(1 for r in rows if r["difficulty_tier"] == t
                    and r["strategy"] == s and r["success"] and r["use_for_reward"])
            cells += f"{n:>12}"
        print(f"{t:10s}{cells}")
 
    _rule("Columns")
    keys = sorted({k for r in rows for k in r})
    shared = sorted(k for k in keys if all(k in r for r in rows))
    partial = sorted(set(keys) - set(shared))
    print(f"   {len(shared)} columns on every row:")
    print("      " + ", ".join(shared))
    if partial:
        print(f"   {len(partial)} strategy-specific columns (passed through):")
        for k in partial:
            owners = sorted({r["strategy"] for r in rows if k in r})
            print(f"      {k}  ({', '.join(owners)})")
 
 
def write_rows(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
 
 
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="../data/processed/""6_budgets_ALL_tasks_decontaminated_batch/episodes_flagged.jsonl")
    ap.add_argument("--metadata", default="../data/processed/""final_annotation_difficulty_tiers/task_metadata.jsonl")
    ap.add_argument("--risk", default="../data/processed/task_list/task_risk_levels.jsonl")
    ap.add_argument("--out", default="../data/processed/""6_budgets_ALL_tasks_decontaminated_batch/final_episodes.jsonl")
    ap.add_argument("--split", action="store_true")
    ap.add_argument("--keep-steps", action="store_true")
    # ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
 
    # if args.self_test:
    #     self_test()
    #     return
 
    episodes = load_jsonl(args.episodes)
    meta = {r["task_id"]: r for r in load_jsonl(args.metadata)}
    risk = {r["task_id"]: r for r in load_jsonl(args.risk)}
 
    required = {"use_for_reward", "use_for_routing", "risk_level"}
    if episodes and not required.issubset(episodes[0]):
        raise SystemExit(
            "episodes are missing the contamination columns "
            f"({sorted(required - set(episodes[0]))}). Run flag_episodes.py first.")
 
    rows, report = join(episodes, meta, risk, keep_steps=args.keep_steps)
    summarise(rows, report)
 
    out = Path(args.out)
    write_rows(rows, out)
    print(f"\n[join] wrote {len(rows)} rows -> {out}")
 
    if args.split:
        for s in sorted({r["strategy"] for r in rows}):
            view = out.with_name(f"{out.stem}_{str(s).lower()}{out.suffix}")
            sub = [r for r in rows if r["strategy"] == s]
            write_rows(sub, view)
            print(f"[join] wrote {len(sub):>5} rows -> {view.name}  (view)")