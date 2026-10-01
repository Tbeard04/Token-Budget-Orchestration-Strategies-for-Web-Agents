"""
cohen_kappa.py - agreement between the hand-labelled gold set and the LLMannotator, for the difficulty rubric used
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agreement import confusion, kappa, landis_koch, pearson, spearman
from rubric import DIMENSIONS, TIERS, tier_of

#scores for the dimensions
SCORES = [0, 1, 2]


#Loading and pairing
def load_jsonl(path: str) -> list[dict]:
    #rows is a list of dictionaries
    rows = []
    #loop through the lines in the path
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        #check if the line is not empty
        if line:
            #try to load the line as a JSON object
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows

#function to build the pairs between the gold and predicted rows
def build_pairs(gold_rows: list, pred_rows: list) -> tuple[list, dict]:
    #Match gold and predicted rows on task_id
    pred_by_id = {r.get("task_id"): r for r in pred_rows}

    #create a list to store the pairs, blank rows, and missing rows
    pairs, blank, missing = [], [], []
    #for each gold row in the gold rows
    for g in gold_rows:
        tid = g.get("task_id")
        #if any of the dimensions are None
        if any(g.get(d) is None for d in DIMENSIONS):
            #add the task id to the blank rows
            blank.append(tid)
            continue
        #get the predicted row by task id
        p = pred_by_id.get(tid)
        #if the predicted row is not found
        if p is None:
            #add the task id to the missing rows
            missing.append(tid)
            continue
        #add the pair to the pairs list
        pairs.append({
            "task_id": tid,
            "site": g.get("site") or p.get("site", ""),
            "intent": g.get("intent") or p.get("intent", ""),
            "gold": {d: int(g[d]) for d in DIMENSIONS},
            "pred": {d: int(p[d]) for d in DIMENSIONS},
            "exemplar": bool(p.get("gold_exemplar")),
            "heldout": bool(p.get("gold_heldout")),
            "confidence": p.get("confidence"),
        })

    #create a dictionary to store the diagnostics
    diag = {
        "gold_rows": len(gold_rows),
        "gold_complete": len(gold_rows) - len(blank),
        "blank_gold_rows": blank,
        "missing_from_pred": missing,
        "matched": len(pairs),
    }
    return pairs, diag


# Scoring one subset
def score(pairs: list) -> dict:
    out: dict = {"n": len(pairs), "dimensions": {}}
    if not pairs:
        return out

    #for each dimension in the dimensions
    for d in DIMENSIONS:
        #get the gold scores for the dimension
        g = [p["gold"][d] for p in pairs]
        #get the predicted scores for the dimension
        m = [p["pred"][d] for p in pairs]
        #create a dictionary to store the scores
        out["dimensions"][d] = {
            #calculate the kappa for the dimension
            "kappa": kappa(g, m, SCORES, "none"),
            "kappa_quadratic": kappa(g, m, SCORES, "quadratic"),
            #calculate the kappa for the dimension
            "kappa_linear": kappa(g, m, SCORES, "linear"),
            #calculate the exact agreement for the dimension
            "exact_agreement": sum(1 for x, y in zip(g, m) if x == y) / len(g),
            "within_one": sum(1 for x, y in zip(g, m) if abs(x - y) <= 1) / len(g),
            #calculate the mean of the gold scores
            "mean_gold": sum(g) / len(g),
            #calculate the mean of the predicted scores
            "mean_pred": sum(m) / len(m),
            #calculate the bias of the predicted scores minus the gold scores
            "bias_pred_minus_gold": sum(m) / len(m) - sum(g) / len(g),
            "confusion": confusion(g, m, SCORES),
        }

    #calculate the total scores for the gold and predicted rows
    g_tot = [sum(p["gold"].values()) for p in pairs]
    m_tot = [sum(p["pred"].values()) for p in pairs]
    #calculate the tiers for the gold and predicted rows
    g_tier = [tier_of(t) for t in g_tot]
    #calculate the tiers for the predicted rows
    m_tier = [tier_of(t) for t in m_tot]

    #create a dictionary to store the tier
    out["tier"] = {
        "kappa": kappa(g_tier, m_tier, TIERS, "none"),
        "kappa_quadratic": kappa(g_tier, m_tier, TIERS, "quadratic"),
        "exact_agreement": sum(1 for x, y in zip(g_tier, m_tier) if x == y) / len(pairs),
        "confusion": confusion(g_tier, m_tier, TIERS),
        "gold_distribution": {t: g_tier.count(t) for t in TIERS},
        "pred_distribution": {t: m_tier.count(t) for t in TIERS}
    }
    #create a dictionary to store the rubric total
    out["rubric_total"] = {
        "mean_gold": sum(g_tot) / len(g_tot),
        "mean_pred": sum(m_tot) / len(m_tot),
        "mean_abs_error": sum(abs(x - y) for x, y in zip(g_tot, m_tot)) / len(pairs),
        "pearson": pearson(g_tot, m_tot),
        "spearman": spearman(g_tot, m_tot)
    }

    #for each key and field in the keys and fields
    for key, field in (("mean_dimension_kappa", "kappa"), ("mean_dimension_kappa_quadratic", "kappa_quadratic")):
        #get the scores for the field
        ks = [v[field] for v in out["dimensions"].values() if v[field] is not None]
        #calculate the mean of the scores
        out[key] = sum(ks) / len(ks) if ks else None
    return out


#function to print the kappa
def _k(v) -> str:
    return "  n/a" if v is None else f"{v:>5.2f}"

#function to print the pearson correlation coefficient
def _r(v) -> str:
    return "n/a" if v is None else f"{v:.2f}"

#function to print the subset
def print_subset(name: str, s: dict) -> None:
    print(f"\n{'=' * 74}\n{name}  (n={s['n']})\n{'=' * 74}")
    #if the number of matched tasks is 0
    if not s["n"]:
        #print the number of matched tasks
        print("no matched tasks")
        return

    print(f"{'dimension':22s} {'kappa':>6} {'quad':>6} {'exact':>7} {'+-1':>6} "
          f"{'bias':>7}  interpretation (quadratic)")
    #loop through the dimensions
    for d in DIMENSIONS:
        #get the scores for the dimension
        v = s["dimensions"][d]
        #print the dimension
        print(f"{d:22s} {_k(v['kappa'])} {_k(v['kappa_quadratic'])} "
              f"{v['exact_agreement']:>6.0%} {v['within_one']:>6.0%} "
              f"{v['bias_pred_minus_gold']:>+7.2f}  {landis_koch(v['kappa_quadratic'])}")

    #get the tier scores
    t = s["tier"]
    #print the difficulty tier
    print(f"\n{'difficulty tier':22s} {_k(t['kappa'])} {_k(t['kappa_quadratic'])} "
          f"{t['exact_agreement']:>6.0%} {landis_koch(t['kappa'])}")
    #print the mean of 4 dimensions
    print(f"{'mean of 4 dimensions':22s} {_k(s['mean_dimension_kappa'])} "
          f"{_k(s['mean_dimension_kappa_quadratic'])}")
    #get the rubric total
    r = s["rubric_total"]
    #print the rubric total
    print(f"\nrubric total 0-8:  human mean {r['mean_gold']:.2f}   "
          f"model mean {r['mean_pred']:.2f} MAE {r['mean_abs_error']:.2f}   "
          f"r={_r(r['pearson'])} rho={_r(r['spearman'])}")

    #print the tier confusion
    print("\ntier confusion (rows = human, cols = model)")
    #loop through the tiers
    print(" " + "".join(f"{c:>10}" for c in TIERS))
    for i, row in enumerate(t["confusion"]):
        print(f"{TIERS[i]:10s}" + "".join(f"{v:>10}" for v in row))


#function to print the confusions
def print_confusions(s: dict) -> None:
    #print the heading
    print(f"\n{'=' * 74}\nPer-dimension confusion (rows = human, cols = model)\n{'=' * 74}")
    #for each dimension in the dimensions
    for d in DIMENSIONS:
        #print the dimension
        print(f"\n{d}")
        print(" " + "".join(f"{j:>7}" for j in SCORES))
        for i, row in enumerate(s["dimensions"][d]["confusion"]):
            print(f"   {i:>4}" + "".join(f"{v:>7}" for v in row))


#function to print the disagreements
def print_disagreements(pairs: list, limit: int) -> None:
    #create a list to store the scored pairs
    scored = []
    for p in pairs:
        #calculate the gap between the gold and predicted scores
        gap = sum(abs(p["gold"][d] - p["pred"][d]) for d in DIMENSIONS)
        #if the gap is not 0
        if gap:
            scored.append((gap, p))
    #sort the scored pairs by the gap
    scored.sort(key=lambda x: -x[0])

    print(f"\n{'=' * 74}\nLargest disagreements "
          f"({len(scored)} of {len(pairs)} tasks differ)\n{'=' * 74}")
    #loop through the scored pairs
    for gap, p in scored[:limit]:
        #get the sub
        sub = "exemplar" if p["exemplar"] else ("held out" if p["heldout"] else "")
        #print the task id, site, gap and sub
        print(f"\n task {p['task_id']:>4}  {p['site']:15s} gap {gap}  {sub}")
        #print the intent
        print(f"{p['intent'][:90]}")
        #loop through the dimensions
        for d in DIMENSIONS:
            #get the gold and predicted scores
            g, m = p["gold"][d], p["pred"][d]
            if g != m:
                #print the dimension, human score and model score
                print(f" {d:22s} human {g}  model {m}")


# Main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold")
    ap.add_argument("--pred")
    ap.add_argument("--out", default=None)
    ap.add_argument("--show", type=int, default=15)
    ap.add_argument("--confusion", action="store_true")
    args = ap.parse_args()

    if not args.gold or not args.pred:
        ap.error("--gold and --pred are required")

    gold_rows = load_jsonl(args.gold)
    pred_rows = load_jsonl(args.pred)
    pairs, diag = build_pairs(gold_rows, pred_rows)

    print(f"[kappa] gold file: {args.gold}")
    print(f"[kappa] pred file: {args.pred}")
    print(f"[kappa] gold rows: {diag['gold_rows']}  "
          f"complete: {diag['gold_complete']}  matched: {diag['matched']}")
    if diag["blank_gold_rows"]:
        print(f"[kappa] SKIPPED {len(diag['blank_gold_rows'])} gold rows with "
              f"unfilled scores: {diag['blank_gold_rows'][:20]}")
    if diag["missing_from_pred"]:
        print(f"[kappa] {len(diag['missing_from_pred'])} gold tasks absent from "
              f"the annotator output: {diag['missing_from_pred'][:20]}")

    #function to get the heldout and exemplar pairs
    heldout = [p for p in pairs if p["heldout"]]
    #function to get the exemplar pairs
    exemplar = [p for p in pairs if p["exemplar"]]
    #function to get the unmarked pairs
    unmarked = [p for p in pairs if not p["heldout"] and not p["exemplar"]]

    #create a dictionary to store the report
    report = {
        "gold_file": args.gold,
        "pred_file": args.pred,
        "diagnostics": diag,
        "all": score(pairs),
    }
    print_subset("ALL matched gold tasks", report["all"])

    #check if the heldout or exemplar pairs are not empty
    if heldout or exemplar:
        #score the heldout pairs
        report["heldout"] = score(heldout)
        report["exemplar"] = score(exemplar)
        print_subset("HELD OUT", report["heldout"])
        print_subset("EXEMPLARS", report["exemplar"])
        if unmarked:
            report["unmarked"] = score(unmarked)
            print_subset("UNMARKED - no exemplar flag in the pred file", report["unmarked"])

    if args.confusion:
        print_confusions(report["all"])
    if args.show:
        print_disagreements(pairs, args.show)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\n[kappa] wrote {out_path}")

if __name__ == "__main__":
    main()