"""
probe.py - run Strategy A over several tasks per site and summarise.

Purpose: gather enough episodes to (a) see the spread of cost and success
across sites, and (b) decide the final token budget ladder.

Imports wa_env and strategy_a rather than duplicating them, so agent config,
prompt, AXTree filtering and budget logic stay in one place.

Tasks are sampled RANDOMLY with a fixed seed. WebArena tasks are template
generated, so consecutive ids are often variants of the same question -
taking the first n per site would give an unrepresentative sample.

Run:
    python probe.py                      # 5 tasks/site, cap from smoke_a
    python probe.py --n 5 --cap 32000    # recommended first run
    python probe.py --sites reddit       # one site only
    python probe.py --seed 7             # different sample
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import wa_env as W
import strategy_a as S


def sample_tasks(n: int, sites: list[str], seed: int) -> dict[str, list[int]]:
    """Randomly sample n single-site task ids per site, reproducibly."""
    pools = W.single_site_tasks(sites)
    rng = random.Random(seed)
    out = {}
    for site in sites:
        pool = pools.get(site, [])
        if not pool:
            print(f"[probe] WARNING: no single-site tasks found for '{site}'")
            out[site] = []
            continue
        k = min(n, len(pool))
        if k < n:
            print(f"[probe] note: only {k} tasks available for '{site}'")
        out[site] = sorted(rng.sample(pool, k))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5, help="tasks per site")
    ap.add_argument("--cap", type=int, default=None, help="override token cap")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sites", nargs="+", default=W.SITES)
    ap.add_argument("--out", default="../data/raw/probe_results.jsonl")
    args = ap.parse_args()

    cap = args.cap or S.DEFAULT_BUDGET
    print(f"[probe] token cap  : {cap}")
    print(f"[probe] max steps  : {W.MAX_STEPS}")
    print(f"[probe] seed       : {args.seed}")

    plan = sample_tasks(args.n, args.sites, args.seed)
    print("[probe] sampled tasks:")
    for site, ids in plan.items():
        print(f"   {site:16s} {ids}")

    out_path = Path(args.out)
    results: list[dict] = []

    # Append as we go so a crash never loses completed episodes.
    with out_path.open("w") as fh:
        for site, ids in plan.items():
            for tid in ids:
                try:
                    rec = S.run_episode(tid, cap)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"\n!! task {tid} failed: {type(e).__name__}: {e}")
                    rec = {"site": site, "task_id": tid,
                           "error": f"{type(e).__name__}: {e}"}
                results.append(rec)
                fh.write(json.dumps(rec) + "\n")
                fh.flush()

    ok = [r for r in results if "error" not in r]

    # ---- per episode --------------------------------------------------------
    print(f"\n\n{'=' * 84}\nPER-EPISODE\n{'=' * 84}")
    print(f"{'site':16s}{'task':>6s}{'ok':>7s}{'steps':>7s}{'tokens':>9s}"
          f"{'tok/step':>10s}{'secs':>8s}  reason")
    for r in results:
        if "error" in r:
            print(f"{r['site']:16s}{r['task_id']:>6}{'ERR':>7s}"
                  f"{'-':>7s}{'-':>9s}{'-':>10s}{'-':>8s}  {r['error'][:30]}")
        else:
            per = r["total_tokens"] / max(r["steps"], 1)
            print(f"{r['site']:16s}{r['task_id']:>6}{str(r['success']):>7s}"
                  f"{r['steps']:>7}{r['total_tokens']:>9}{per:>10.0f}"
                  f"{r['wall_clock_seconds']:>8}  {r['termination_reason']}")

    # ---- per site -----------------------------------------------------------
    print(f"\n{'=' * 84}\nPER-SITE MEDIANS\n{'=' * 84}")
    print(f"{'site':16s}{'n':>4s}{'success':>9s}{'tokens':>9s}{'steps':>8s}"
          f"{'tok/step':>10s}{'secs':>8s}")
    for site in args.sites:
        rs = [r for r in ok if r.get("site") == site]
        if not rs:
            print(f"{site:16s}{0:>4}   (no usable episodes)")
            continue
        sr = sum(1 for r in rs if r["success"]) / len(rs)
        print(f"{site:16s}{len(rs):>4}{sr:>8.0%}"
              f"{statistics.median(r['total_tokens'] for r in rs):>9.0f}"
              f"{statistics.median(r['steps'] for r in rs):>8.1f}"
              f"{statistics.median(r['total_tokens'] / max(r['steps'],1) for r in rs):>10.0f}"
              f"{statistics.median(r['wall_clock_seconds'] for r in rs):>8.1f}")

    # ---- overall ------------------------------------------------------------
    if ok:
        toks = sorted(r["total_tokens"] for r in ok)
        print(f"\n{'=' * 84}\nOVERALL TOKEN DISTRIBUTION (for setting the ladder)"
              f"\n{'=' * 84}")
        print(f"   episodes           : {len(ok)}")
        print(f"   min                : {toks[0]}")
        print(f"   25th percentile    : {toks[max(0, len(toks)//4 - 1)]}")
        print(f"   median             : {statistics.median(toks):.0f}")
        print(f"   75th percentile    : {toks[min(len(toks)-1, 3*len(toks)//4)]}")
        print(f"   max                : {toks[-1]}")
        succ = [r["total_tokens"] for r in ok if r["success"]]
        if succ:
            print(f"   median of SUCCESSES: {statistics.median(succ):.0f}   "
                  f"(n={len(succ)})")

    # ---- termination reasons ------------------------------------------------
    print(f"\n{'=' * 84}\nTERMINATION REASONS\n{'=' * 84}")
    reasons: dict[str, int] = {}
    for r in results:
        key = r.get("termination_reason", "ERROR")
        reasons[key] = reasons.get(key, 0) + 1
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        print(f"   {k:24s} {v}")

    print(f"\nWrote {len(results)} episodes to {out_path}")


if __name__ == "__main__":
    main()