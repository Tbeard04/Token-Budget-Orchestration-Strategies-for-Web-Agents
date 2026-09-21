"""
flag_episodes.py - decide which episodes Strategy C may learn a reward from.
"""
from __future__ import annotations
 
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
 
RISK_LEVELS = ["read_only", "idempotent", "non_idempotent"]
 
 
def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows

def demanding(meta: dict | None) -> bool:
    #if the metadata is not provided
    if not meta:
        #no metadata: fall back to flagging everything
        return True
    #if the interaction is 2 or the difficulty tier is Medium or Hard
    return (meta.get("interaction") == 2
            or meta.get("difficulty_tier") in ("Medium", "Hard"))


def flag(episodes: list[dict], risk: dict, lenient_idempotent: bool = False, pre_steps: int = 1, metadata: dict | None = None) -> list[dict]:

    #if the metadata is not provided, use an empty dictionary
    metadata = metadata or {}
    #iterate over the episodes and remember the input order
    for i, e in enumerate(episodes):
        e["_i"] = i
 
    #create a dictionary to store the episodes by strategy and task_id
    groups: dict[tuple, list[dict]] = defaultdict(list)
    #iterate over the episodes and add them to the dictionary
    for e in episodes:
        groups[(e.get("strategy"), e.get("task_id"))].append(e)
 
    #iterate over the episodes and add them to the dictionary
    for (strategy, task_id), eps in groups.items():
        #get the level
        level = risk.get(task_id, "unknown")
        #sort the episodes by timestamp and input order
        eps.sort(key=lambda e: (e.get("timestamp", ""), e["_i"]))
        #get the first success
 
        first = next((i for i, e in enumerate(eps) if e.get("success")), None)
        fs_budget = eps[first].get("budget_level") if first is not None else None
        fs_steps = eps[first].get("steps") if first is not None else None
 
        #if the level is idempotent or non_idempotent and the first success is not None and the steps are less than or equal to the pre_steps and the metadata is demanding
        suspect = (level in ("idempotent", "non_idempotent")
                   and first is not None
                   and (fs_steps or 0) <= pre_steps
                   and demanding(metadata.get(task_id)))
 
        exempt = level == "read_only" or (lenient_idempotent and level == "idempotent")
        #iterate over the episodes and add the metadata
        for idx, e in enumerate(eps):
            after = first is not None and idx > first
            excluded = after and not exempt

            #add the metadata to the episode
            e["risk_level"] = level
            e["episode_index"] = idx
            e["first_success_index"] = first
            e["first_success_budget"] = fs_budget
            e["suspect_pre_existing"] = suspect
            e["use_for_routing"] = True
            e["use_for_reward"] = not excluded
            if not excluded:
                e["contamination_reason"] = None
            elif level == "idempotent":
                e["contamination_reason"] = "idempotent_post_first_success"
            else:
                e["contamination_reason"] = "post_first_success"
 
    for e in episodes:
        del e["_i"]
    return episodes


# reporting
#print the title
def _rule(title: str) -> None:
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")
 
#print the report
def report(episodes: list[dict], pre_steps: int) -> None:
    n = len(episodes)
    by_strategy = defaultdict(list)
    for e in episodes:
        by_strategy[e.get("strategy")].append(e)
 
    _rule("Reward eligibility")
    print(f"{'strategy':10s}{'episodes':>10}{'usable':>10}{'excluded':>10}"
          f"{'excluded %':>12}")
    for s in sorted(by_strategy):
        eps = by_strategy[s]
        bad = sum(1 for e in eps if not e["use_for_reward"])
        print(f"{str(s):10s}{len(eps):>10}{len(eps) - bad:>10}{bad:>10}"
              f"{bad / len(eps):>11.1%}")
    bad = sum(1 for e in episodes if not e["use_for_reward"])
    print(f"{'ALL':10s}{n:>10}{n - bad:>10}{bad:>10}{bad / n:>11.1%}")
    print(f"\nuse_for_routing: {sum(1 for e in episodes if e['use_for_routing'])}"
          f" of {n} (all episodes, by design)")
 
    _rule("Exclusions by reason")
    reasons = Counter(e["contamination_reason"] for e in episodes if e["contamination_reason"])
    if not reasons:
        print("none")
    for reason, c in reasons.most_common():
        print(f"{reason:32s} {c:>5}")
 
    #print the episodes by risk level
    _rule("Episodes by risk level")
    print(f"{'risk level':18s}{'episodes':>10}{'usable':>10}{'excluded':>10}")
    for lv in RISK_LEVELS + ["unknown"]:
        eps = [e for e in episodes if e["risk_level"] == lv]
        if not eps:
            continue
        bad = sum(1 for e in eps if not e["use_for_reward"])
        print(f"{lv:18s}{len(eps):>10}{len(eps) - bad:>10}{bad:>10}")
 
    #print the successes
    _rule("Successes (what the reward signal is actually made of)")
    print(f"{'strategy':10s}{'successes':>11}{'usable':>9}{'discarded':>11}")
    for s in sorted(by_strategy):
        su = [e for e in by_strategy[s] if e.get("success")]
        bad = sum(1 for e in su if not e["use_for_reward"])
        print(f"{str(s):10s}{len(su):>11}{len(su) - bad:>9}{bad:>11}")
 
    suspects = sorted({(e["task_id"], e["risk_level"]) for e in episodes if e["suspect_pre_existing"]})
    _rule(f"Suspect pre-existing state (first success took <= {pre_steps} step)")
    print(f"{len(suspects)} task/level pairs. These are NOT excluded - a task "
          f"that genuinely\ntakes one step looks identical. Report them as a "
          f"limitation, or exclude\nthem in a sensitivity analysis.\n")
    for tid, lv in suspects[:20]:
        print(f"task {tid:>4}  {lv}")
    if len(suspects) > 20:
        print(f"and {len(suspects) - 20} more")


#main 
def main() -> None:
    ap = argparse.ArgumentParser()
    #add the episodes
    ap.add_argument("--episodes", nargs="+", default=[
        "../data/processed/6_budgets_ALL_tasks_decontaminated_batch/batch_strategy_A/strategy_a.jsonl",
        "../data/processed/6_budgets_ALL_tasks_decontaminated_batch/batch_strategy_B/strategy_b.jsonl",])
    ap.add_argument("--risk", default="../data/processed/task_list/task_risk_levels.jsonl")
    ap.add_argument("--metadata", default="../data/processed/" "final_annotation_difficulty_tiers/task_metadata.jsonl")
    ap.add_argument("--out", default="../data/processed/" "6_budgets_ALL_tasks_decontaminated_batch/episodes_flagged.jsonl")
    ap.add_argument("--lenient-idempotent", action="store_true")
    ap.add_argument("--pre-steps", type=int, default=1)
    ap.add_argument("--example", type=int, default=None)
    # ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
 
    # if args.self_test:
    #     self_test()
    #     return
 
    risk_rows = load_jsonl(args.risk)
    risk = {r["task_id"]: r["risk_level"] for r in risk_rows}
    print(f"[flag] risk levels : {len(risk)} tasks from {args.risk}")
 
    episodes, skipped = [], 0
    for path in args.episodes:
        rows = load_jsonl(path)
        errs = [r for r in rows if "error" in r]
        clean = [r for r in rows if "error" not in r]
        skipped += len(errs)
        strat = Counter(r.get("strategy") for r in clean)
        print(f"[flag] {Path(path).name:22s} {len(clean):>5} episodes"
              f"  strategies={dict(strat)}"
              + (f"  ({len(errs)} error rows skipped)" if errs else ""))
        episodes.extend(clean)
 
    if not episodes:
        raise SystemExit("no episodes loaded")
 
    missing = sorted({e["task_id"] for e in episodes} - set(risk))
    if missing:
        print(f"[flag] WARNING: {len(missing)} task ids have no risk level and "
              f"are treated as unknown (never exempt): {missing[:15]}")
 
    meta = {}
    if Path(args.metadata).exists():
        meta = {m["task_id"]: m for m in load_jsonl(args.metadata)}
        print(f"[flag] metadata   : {len(meta)} tasks from {args.metadata}")
    else:
        print(f"[flag] WARNING: no metadata at {args.metadata} - every "
              f"one-step first success will be marked suspect, including "
              f"genuinely one-step tasks")
 
    if args.lenient_idempotent:
        print("[flag] --lenient-idempotent: idempotent tasks treated as clean")
 
    flag(episodes, risk, args.lenient_idempotent, args.pre_steps, meta)
 
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")
 
    report(episodes, args.pre_steps)
    print(f"\n[flag] wrote {len(episodes)} episodes to {out}")
    if skipped:
        print(f"[flag] {skipped} error rows were skipped and are NOT in the output")