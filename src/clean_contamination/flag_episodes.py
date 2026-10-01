"""
flag_episodes.py - decide which episodes Strategy C may learn a reward from
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

#risk levels
RISK_LEVELS = ["read_only", "idempotent", "non_idempotent"]


#function to load the jsonl file
def load_jsonl(path: str | Path) -> list[dict]:
    #rows is a list of dictionaries
    rows = []
    #for each line in the file
    for line in Path(path).read_text().splitlines():
        #strip the line
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows

#function to check if the task is demanding
def demanding(meta: dict | None) -> bool:
    #if the metadata is not provided
    if not meta:
        #no metadata then fall back to flagging everything
        return True
    #if the interaction is 2 or the difficulty tier is Medium or Hard
    return (meta.get("interaction") == 2
            or meta.get("difficulty_tier") in ("Medium", "Hard"))

#function to flag the episodes
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

    #print the reward eligibility
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

    #print the exclusions by reason
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
    #print the suspect pre-existing state
    _rule(f"Suspect pre-existing state (first success took <= {pre_steps} step)")
    for tid, lv in suspects[:20]:
        print(f"task {tid:>4}  {lv}")
    if len(suspects) > 20:
        print(f"and {len(suspects) - 20} more")


#main 
def main() -> None:
    ap = argparse.ArgumentParser()
    #add the episodes
    ap.add_argument("--episodes", nargs="+", default=[
        "../data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_A/strategy_a.jsonl",
        "../data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_B/strategy_b.jsonl",])
    ap.add_argument("--risk", default="../data/processed/task_list/task_risk_levels.jsonl")
    ap.add_argument("--metadata", default="../data/processed/" "final_annotation_difficulty_tiers/task_metadata.jsonl")
    ap.add_argument("--out", default="../data/processed/" "6_budgets_ALL_tasks_decontaminated_batch/episodes_flagged.jsonl")
    ap.add_argument("--lenient-idempotent", action="store_true")
    ap.add_argument("--pre-steps", type=int, default=1)
    ap.add_argument("--example", type=int, default=None)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    #load the risk levels
    risk_rows = load_jsonl(args.risk)
    risk = {r["task_id"]: r["risk_level"] for r in risk_rows}
    print(f"[flag] risk levels : {len(risk)} tasks from {args.risk}")

    #load the episodes
    episodes, skipped = [], 0
    #for each path in the episodes
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

    #get the missing task ids
    missing = sorted({e["task_id"] for e in episodes} - set(risk))
    #if there are missing task ids
    if missing:
        print(f"[flag] WARNING: {len(missing)} task ids have no risk level and "
              f"are treated as unknown (never exempt): {missing[:15]}")

    #load the metadata
    meta = {}
    #if the metadata file exists
    if Path(args.metadata).exists():
        #load the metadata
        meta = {m["task_id"]: m for m in load_jsonl(args.metadata)}
        print(f"[flag] metadata   : {len(meta)} tasks from {args.metadata}")
    else:
        print(f"[flag] WARNING: no metadata at {args.metadata} - every "
              f"one-step first success will be marked suspect, including "
              f"genuinely one-step tasks")

    #print the lenient idempotent flag
    if args.lenient_idempotent:
        print("[flag] --lenient-idempotent: idempotent tasks treated as clean")

    #flag the episodes
    flag(episodes, risk, args.lenient_idempotent, args.pre_steps, meta)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for e in episodes:
            f.write(json.dumps(e) + "\n")

    #report the episodes
    report(episodes, args.pre_steps)
    #print the number of episodes written
    print(f"\n[flag] wrote {len(episodes)} episodes to {out}")
    #if there are skipped error rows
    if skipped:
        #print the number of error rows skipped
        print(f"[flag] {skipped} error rows were skipped and are NOT in the output")


def self_test() -> None:
    def ep(strategy, task_id, ts, budget, success, steps=3):
        return {"strategy": strategy, "task_id": task_id, "timestamp": ts, "budget_level": budget, "success": success, "steps": steps, "total_tokens": steps * 1000, "goal": "g"}

    #add the risk levels for self test
    risk = {1: "non_idempotent", 2: "read_only", 3: "idempotent", 4: "non_idempotent", 5: "non_idempotent"}

    eps = [
        #add the episodes for self test
        ep("A", 1, "T3", 8000, True), ep("A", 1, "T1", 4000, False),
        ep("A", 1, "T2", 16000, True), ep("A", 1, "T4", 2000, True),
        # task 2: read_only, two successes, nothing dropped
        ep("A", 2, "T1", 4000, True), ep("A", 2, "T2", 8000, True),
        # task 3: idempotent -> post-first-success dropped, own reason
        ep("A", 3, "T1", 4000, True), ep("A", 3, "T2", 8000, True),
        # task 4: A succeeds early, B succeeds late. B must not be punished
        # for A's success -> proves grouping includes strategy.
        ep("A", 4, "T1", 4000, True), ep("A", 4, "T2", 8000, True),
        ep("B", 4, "T1", 4000, False), ep("B", 4, "T2", 8000, True),
        # task 5: no success at all -> nothing dropped
        ep("A", 5, "T1", 4000, False), ep("A", 5, "T2", 8000, False),
    ]
    flag(eps, risk)

    #function to get the episode by strategy, task_id and budget_level
    def get(s, t, b):
        #return the next episode that matches the strategy, task_id and budget_level
        return next(e for e in eps if e["strategy"] == s and e["task_id"] == t and e["budget_level"] == b)

    #get the episode by strategy, task_id and budget_level
    #assert the episode index is correct
    assert get("A", 1, 4000)["episode_index"] == 0
    assert get("A", 1, 16000)["episode_index"] == 1
    assert get("A", 1, 8000)["episode_index"] == 2 
    assert get("A", 1, 2000)["episode_index"] == 3
    #assert the use for reward is correct
    assert get("A", 1, 4000)["use_for_reward"] is True
    assert get("A", 1, 16000)["use_for_reward"] is True
    assert get("A", 1, 8000)["use_for_reward"] is False
    assert get("A", 1, 2000)["use_for_reward"] is False
    #assert the contamination reason is correct
    assert get("A", 1, 8000)["contamination_reason"] == "post_first_success"
    #assert the first success budget is correct
    assert get("A", 1, 16000)["first_success_budget"] == 16000

    #read_only is exempt
    assert all(e["use_for_reward"] for e in eps if e["task_id"] == 2)

    #idempotent is NOT exempt, and carries its own reason
    assert get("A", 3, 8000)["use_for_reward"] is False
    assert get("A", 3, 8000)["contamination_reason"] == "idempotent_post_first_success"

    #per-strategy grouping: B's episodes survive A's success
    assert get("A", 4, 8000)["use_for_reward"] is False
    assert get("B", 4, 4000)["use_for_reward"] is True
    assert get("B", 4, 8000)["use_for_reward"] is True
    assert get("B", 4, 8000)["first_success_index"] == 1

    #no success --> nothing dropped
    assert all(e["use_for_reward"] for e in eps if e["task_id"] == 5)
    assert all(e["first_success_index"] is None for e in eps if e["task_id"] == 5)

    #routing always survives
    assert all(e["use_for_routing"] for e in eps)

    #lenient mode spares idempotent only
    #add the episodes for self test
    eps2 = [ep("A", 3, "T1", 4000, True), ep("A", 3, "T2", 8000, True), ep("A", 1, "T1", 4000, True), ep("A", 1, "T2", 8000, True)]
    #flag the episodes
    flag(eps2, risk, lenient_idempotent=True)
    assert all(e["use_for_reward"] for e in eps2 if e["task_id"] == 3)
    #assert the use for reward is correct
    assert eps2[3]["use_for_reward"] is False

    #suspect_pre_existing fires on a one-step first success on a writing task
    eps3 = [ep("A", 1, "T1", 4000, False, steps=2), ep("A", 1, "T2", 8000, True, steps=1)]
    flag(eps3, risk)
    assert all(e["suspect_pre_existing"] for e in eps3)
    eps4 = [ep("A", 2, "T1", 4000, True, steps=1)]   # read_only never suspect
    flag(eps4, risk)
    assert not eps4[0]["suspect_pre_existing"]

    #add the metadata for self test
    cheap = {1: {"task_id": 1, "interaction": 1, "difficulty_tier": "Easy"}}
    eps6 = [ep("A", 1, "T1", 4000, True, steps=1)]
    flag(eps6, risk, metadata=cheap)
    assert not eps6[0]["suspect_pre_existing"]

    #add the metadata for self test
    hard = {1: {"task_id": 1, "interaction": 2, "difficulty_tier": "Easy"}}
    eps7 = [ep("A", 1, "T1", 4000, True, steps=1)]
    flag(eps7, risk, metadata=hard)
    assert eps7[0]["suspect_pre_existing"]

    #a Medium/Hard tier alone is enough, even at interaction 1
    med = {1: {"task_id": 1, "interaction": 1, "difficulty_tier": "Medium"}}
    eps8 = [ep("A", 1, "T1", 4000, True, steps=1)]
    flag(eps8, risk, metadata=med)
    assert eps8[0]["suspect_pre_existing"]

    #add the metadata for self test
    eps9 = [ep("A", 1, "T1", 4000, True, steps=1)]
    flag(eps9, risk, metadata={})
    assert eps9[0]["suspect_pre_existing"]


    #assert the demanding function is correct
    assert demanding(None) is True
    assert demanding({"interaction": 0, "difficulty_tier": "Easy"}) is False

    # unknown risk level is never exempt
    eps5 = [ep("A", 99, "T1", 4000, True), ep("A", 99, "T2", 8000, True)]
    flag(eps5, risk)
    assert eps5[0]["risk_level"] == "unknown"
    assert eps5[1]["use_for_reward"] is False

    print("self-test: all assertions passed")

if __name__ == "__main__":
    main()