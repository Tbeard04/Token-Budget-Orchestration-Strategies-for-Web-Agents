"""
cohen_kappa.py - agreement between the hand-labelled gold set and the LLMannotator, for the difficulty rubric used
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from agreement import confusion, kappa, landis_koch, pearson, spearman
from rubric import DIMENSIONS, TIERS, tier_of

SCORES = [0, 1, 2]


# Loading and pairing
def load_jsonl(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def build_pairs(gold_rows: list, pred_rows: list) -> tuple[list, dict]:
    #Match gold and predicted rows on task_id
    pred_by_id = {r.get("task_id"): r for r in pred_rows}

    pairs, blank, missing = [], [], []
    for g in gold_rows:
        tid = g.get("task_id")
        if any(g.get(d) is None for d in DIMENSIONS):
            blank.append(tid)
            continue
        p = pred_by_id.get(tid)
        if p is None:
            missing.append(tid)
            continue
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

    for d in DIMENSIONS:
        g = [p["gold"][d] for p in pairs]
        m = [p["pred"][d] for p in pairs]
        out["dimensions"][d] = {
            "kappa": kappa(g, m, SCORES, "none"),
            "kappa_quadratic": kappa(g, m, SCORES, "quadratic"),
            "kappa_linear": kappa(g, m, SCORES, "linear"),
            "exact_agreement": sum(1 for x, y in zip(g, m) if x == y) / len(g),
            "within_one": sum(1 for x, y in zip(g, m) if abs(x - y) <= 1) / len(g),
            "mean_gold": sum(g) / len(g),
            "mean_pred": sum(m) / len(m),
            # systematic drift - kappa can look fine while the model sits half a
            # point above the human throughout, which needs recalibration rather than a rubric rewrite
            "bias_pred_minus_gold": sum(m) / len(m) - sum(g) / len(g),
            "confusion": confusion(g, m, SCORES),
        }

    g_tot = [sum(p["gold"].values()) for p in pairs]
    m_tot = [sum(p["pred"].values()) for p in pairs]
    g_tier = [tier_of(t) for t in g_tot]
    m_tier = [tier_of(t) for t in m_tot]

    out["tier"] = {
        "kappa": kappa(g_tier, m_tier, TIERS, "none"),
        "kappa_quadratic": kappa(g_tier, m_tier, TIERS, "quadratic"),
        "exact_agreement": sum(1 for x, y in zip(g_tier, m_tier) if x == y) / len(pairs),
        "confusion": confusion(g_tier, m_tier, TIERS),
        "gold_distribution": {t: g_tier.count(t) for t in TIERS},
        "pred_distribution": {t: m_tier.count(t) for t in TIERS},
    }
    out["rubric_total"] = {
        "mean_gold": sum(g_tot) / len(g_tot),
        "mean_pred": sum(m_tot) / len(m_tot),
        "mean_abs_error": sum(abs(x - y) for x, y in zip(g_tot, m_tot)) / len(pairs),
        "pearson": pearson(g_tot, m_tot),
        "spearman": spearman(g_tot, m_tot),
    }

    for key, field in (("mean_dimension_kappa", "kappa"),
                       ("mean_dimension_kappa_quadratic", "kappa_quadratic")):
        ks = [v[field] for v in out["dimensions"].values() if v[field] is not None]
        out[key] = sum(ks) / len(ks) if ks else None
    return out


# Printing
def _k(v) -> str:
    return "  n/a" if v is None else f"{v:>5.2f}"


def _r(v) -> str:
    return "n/a" if v is None else f"{v:.2f}"


def print_subset(name: str, s: dict) -> None:
    print(f"\n{'=' * 74}\n{name}  (n={s['n']})\n{'=' * 74}")
    if not s["n"]:
        print("   no matched tasks")
        return

    print(f"{'dimension':22s} {'kappa':>6} {'quad':>6} {'exact':>7} {'+-1':>6} "
          f"{'bias':>7}  interpretation (quadratic)")
    for d in DIMENSIONS:
        v = s["dimensions"][d]
        print(f"{d:22s} {_k(v['kappa'])} {_k(v['kappa_quadratic'])} "
              f"{v['exact_agreement']:>6.0%} {v['within_one']:>6.0%} "
              f"{v['bias_pred_minus_gold']:>+7.2f}  {landis_koch(v['kappa_quadratic'])}")

    t = s["tier"]
    print(f"\n{'difficulty tier':22s} {_k(t['kappa'])} {_k(t['kappa_quadratic'])} "
          f"{t['exact_agreement']:>6.0%}                {landis_koch(t['kappa'])}")
    print(f"{'mean of 4 dimensions':22s} {_k(s['mean_dimension_kappa'])} "
          f"{_k(s['mean_dimension_kappa_quadratic'])}")

    r = s["rubric_total"]
    print(f"\nrubric total 0-8:  human mean {r['mean_gold']:.2f}   "
          f"model mean {r['mean_pred']:.2f}   MAE {r['mean_abs_error']:.2f}   "
          f"r={_r(r['pearson'])}   rho={_r(r['spearman'])}")

    print("\ntier confusion (rows = human, cols = model)")
    print("              " + "".join(f"{c:>10}" for c in TIERS))
    for i, row in enumerate(t["confusion"]):
        print(f"   {TIERS[i]:10s}" + "".join(f"{v:>10}" for v in row))


def print_confusions(s: dict) -> None:
    print(f"\n{'=' * 74}\nPer-dimension confusion (rows = human, cols = model)\n{'=' * 74}")
    for d in DIMENSIONS:
        print(f"\n{d}")
        print("        " + "".join(f"{j:>7}" for j in SCORES))
        for i, row in enumerate(s["dimensions"][d]["confusion"]):
            print(f"   {i:>4}" + "".join(f"{v:>7}" for v in row))


def print_disagreements(pairs: list, limit: int) -> None:
    scored = []
    for p in pairs:
        gap = sum(abs(p["gold"][d] - p["pred"][d]) for d in DIMENSIONS)
        if gap:
            scored.append((gap, p))
    scored.sort(key=lambda x: -x[0])

    print(f"\n{'=' * 74}\nLargest disagreements "
          f"({len(scored)} of {len(pairs)} tasks differ)\n{'=' * 74}")
    for gap, p in scored[:limit]:
        sub = "exemplar" if p["exemplar"] else ("held out" if p["heldout"] else "")
        print(f"\n   task {p['task_id']:>4}  {p['site']:15s} gap {gap}  {sub}")
        print(f"      {p['intent'][:90]}")
        for d in DIMENSIONS:
            g, m = p["gold"][d], p["pred"][d]
            if g != m:
                print(f"      {d:22s} human {g}  model {m}")


def print_methodology_line(report: dict) -> None:
    src = report.get("heldout") or report["all"]
    label = "held-out" if report.get("heldout", {}).get("n") else "all-task"
    if not src["n"]:
        return
    print(f"\n{'=' * 74}\nFor the methodology\n{'=' * 74}")
    print(f"On the {label} subset (n={src['n']}), quadratic-weighted Cohen's kappa "
          f"between the")
    print(f"human labels and the LLM annotator was "
          f"{_k(src['mean_dimension_kappa_quadratic']).strip()} averaged over the "
          f"four rubric")
    print(f"dimensions ({landis_koch(src['mean_dimension_kappa_quadratic'])}), and "
          f"{_k(src['tier']['kappa']).strip()} on the three-band difficulty tier")
    print(f"({landis_koch(src['tier']['kappa'])}). Per-dimension figures are in the "
          f"table above.")


# Main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", help="hand-labelled gold set jsonl")
    ap.add_argument("--pred", help="annotator output jsonl")
    ap.add_argument("--out", default=None, help="write the full report as JSON")
    ap.add_argument("--show", type=int, default=15,
                    help="how many disagreements to print")
    ap.add_argument("--confusion", action="store_true",
                    help="also print per-dimension confusion matrices")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return
    if not args.gold or not args.pred:
        ap.error("--gold and --pred are required")

    gold_rows = load_jsonl(args.gold)
    pred_rows = load_jsonl(args.pred)
    pairs, diag = build_pairs(gold_rows, pred_rows)

    print(f"[kappa] gold file : {args.gold}")
    print(f"[kappa] pred file : {args.pred}")
    print(f"[kappa] gold rows : {diag['gold_rows']}  "
          f"complete: {diag['gold_complete']}  matched: {diag['matched']}")
    if diag["blank_gold_rows"]:
        print(f"[kappa] SKIPPED {len(diag['blank_gold_rows'])} gold rows with "
              f"unfilled scores: {diag['blank_gold_rows'][:20]}")
    if diag["missing_from_pred"]:
        print(f"[kappa] {len(diag['missing_from_pred'])} gold tasks absent from "
              f"the annotator output: {diag['missing_from_pred'][:20]}")

    if not pairs:
        print("\nNothing to compare. Fill in the gold set scores first.")
        return

    heldout = [p for p in pairs if p["heldout"]]
    exemplar = [p for p in pairs if p["exemplar"]]
    unmarked = [p for p in pairs if not p["heldout"] and not p["exemplar"]]

    report = {
        "gold_file": args.gold,
        "pred_file": args.pred,
        "diagnostics": diag,
        "all": score(pairs),
    }
    print_subset("ALL matched gold tasks", report["all"])

    if heldout or exemplar:
        report["heldout"] = score(heldout)
        report["exemplar"] = score(exemplar)
        print_subset("HELD OUT - never shown to the annotator (quote this one)",
                     report["heldout"])
        print_subset("EXEMPLARS - shown to the annotator (inflated by design)",
                     report["exemplar"])
        if unmarked:
            report["unmarked"] = score(unmarked)
            print_subset("UNMARKED - no exemplar flag in the pred file",
                         report["unmarked"])
    else:
        print("\n[kappa] the annotator output carries no gold_exemplar / "
              "gold_heldout flags, so the subsets above cannot be split. If the "
              "gold labels were used as exemplars, this figure is inflated.")

    if args.confusion:
        print_confusions(report["all"])
    if args.show:
        print_disagreements(pairs, args.show)
    print_methodology_line(report)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2))
        print(f"\n[kappa] wrote {out_path}")


# Self-test - kappa checked against hand-computed values
def self_test() -> None:
    def expand(m: list[list[int]], labels: list) -> tuple[list, list]:
        a, b = [], []
        for i, row in enumerate(m):
            for j, c in enumerate(row):
                a += [labels[i]] * c
                b += [labels[j]] * c
        return a, b

    a, b = expand([[10, 5, 0], [5, 10, 5], [0, 5, 10]], SCORES)
    k_u = kappa(a, b, SCORES, "none")
    k_l = kappa(a, b, SCORES, "linear")
    k_q = kappa(a, b, SCORES, "quadratic")
    assert abs(k_u - 0.393939) < 1e-4, k_u
    assert abs(k_q - 0.666667) < 1e-4, k_q
    assert k_u < k_l < k_q, (k_u, k_l, k_q)

    # perfect agreement
    a, b = expand([[10, 0, 0], [0, 10, 0], [0, 0, 10]], SCORES)
    assert kappa(a, b, SCORES, "none") == 1.0
    assert kappa(a, b, SCORES, "quadratic") == 1.0

    #independent raters -> kappa 0
    a, b = expand([[25, 25], [25, 25]], [0, 1])
    assert abs(kappa(a, b, [0, 1], "none")) < 1e-9

    #total disagreement is worse than chance
    a, b = expand([[0, 25], [25, 0]], [0, 1])
    assert kappa(a, b, [0, 1], "none") == -1.0

    #both raters constant and identical -> defined as 1.0
    assert kappa([1] * 10, [1] * 10, SCORES, "none") == 1.0
    # one rater constant, the other not -> defined, and no better than chance
    k = kappa([1] * 10, [1] * 5 + [2] * 5, SCORES, "none")
    assert k is not None and k <= 0, k

    #band cut-points shared with the annotator via rubric.py
    assert [tier_of(t) for t in (0, 2, 3, 5, 6, 8)] == \
        ["Easy", "Easy", "Medium", "Medium", "Hard", "Hard"]

    # rank correlation with ties
    assert abs(spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-9
    assert abs(spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-9
    assert abs(spearman([1, 1, 2, 2], [1, 1, 2, 2]) - 1.0) < 1e-9

    # end-to-end on a tiny pair of files
    gold = [{"task_id": 1, "site": "shopping", "intent": "a",
             "pages_to_traverse": 1, "retrieval_type": 0,
             "interaction": 0, "target_locatability": 0},
            {"task_id": 2, "site": "reddit", "intent": "b",
             "pages_to_traverse": 2, "retrieval_type": 2,
             "interaction": 2, "target_locatability": 2},
            {"task_id": 3, "site": "reddit", "intent": "c",
             "pages_to_traverse": None, "retrieval_type": None,
             "interaction": None, "target_locatability": None}]
    pred = [{"task_id": 1, "pages_to_traverse": 1, "retrieval_type": 0,
             "interaction": 0, "target_locatability": 0,
             "gold_exemplar": True, "gold_heldout": False},
            {"task_id": 2, "pages_to_traverse": 2, "retrieval_type": 2,
             "interaction": 2, "target_locatability": 1,
             "gold_exemplar": False, "gold_heldout": True}]
    pairs, diag = build_pairs(gold, pred)
    assert diag["matched"] == 2, diag
    assert diag["blank_gold_rows"] == [3], diag
    s = score(pairs)
    assert s["n"] == 2
    assert s["dimensions"]["pages_to_traverse"]["exact_agreement"] == 1.0
    assert s["dimensions"]["target_locatability"]["exact_agreement"] == 0.5
    assert s["rubric_total"]["mean_abs_error"] == 0.5
    assert sum(1 for p in pairs if p["heldout"]) == 1

    print("self-test: all assertions passed")
    print(f"   kappa unweighted  0.3939 == {k_u:.4f}")
    print(f"   kappa linear             == {k_l:.4f}")
    print(f"   kappa quadratic   0.6667 == {k_q:.4f}")


if __name__ == "__main__":
    main()