from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from scipy.stats import wilcoxon
from matplotlib.lines import Line2D

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from batch_analysis.shared import (load, print_section, wilson, mcnemar_exact, outcome_group, COLOURS, MARKERS, NAMES, TIER_ORDER, OUTCOME_ORDER, OUTCOME_COLOURS)

BUDGET = 32_000
STRATEGIES = ["A", "B", "C"]
SCHEMES = ["pay_as_you_go", "even", "front_loaded", "reactive"]
SCHEME_LABELS = {"pay_as_you_go": "Pay as you go", "even": "Even", "front_loaded": "Front-loaded", "reactive": "Reactive"}

#Loading
#scheme runs + the pay-as-you-go reference from the main batches, on the same tasks
def load_grid(args) -> pd.DataFrame:
    sch = load(args.schemes)
    sch = sch[sch["budget_level"] == BUDGET]
    tasks = set(sch["task_id"])
    parts = [sch]
    for s, path in [("A", args.main_a), ("B", args.main_b), ("C", args.main_c)]:
        if not path:
            continue
        m = load(path)
        m = m[(m["budget_level"] == BUDGET) & (m["task_id"].isin(tasks))].copy()
        m["strategy"] = s
        m["scheme"] = "pay_as_you_go"
        parts.append(m)
    df = pd.concat(parts, ignore_index=True, sort=False)

    #difficulty tier per task: from the task list if given, else from the main-batch files
    tiers = None
    if args.tasks and Path(args.tasks).exists():
        rows = [json.loads(l) for l in Path(args.tasks).read_text().splitlines() if l.strip()]
        tiers = {r["task_id"]: r.get("difficulty_tier") for r in rows}
    elif "difficulty_tier" in df.columns:
        tiers = df.dropna(subset=["difficulty_tier"]).drop_duplicates("task_id").set_index("task_id")["difficulty_tier"].to_dict()
    if tiers:
        df["difficulty_tier"] = df["task_id"].map(tiers)

    #scheme-run mechanics, pulled out of step_log (absent for pay-as-you-go rows)
    def step_stats(steps):
        #main-batch step logs have no scheme fields: not applicable rather than 0
        if not isinstance(steps, list) or not steps or "trimmed" not in steps[0]:
            return pd.Series({"trim_share": np.nan, "high_share": np.nan})
        by_step = {}
        #get the step stats
        for s in steps:
            by_step.setdefault(s["step"], s)
        return pd.Series({"trim_share": np.mean([bool(s.get("trimmed")) for s in by_step.values()]), "high_share": np.mean([s.get("effort") == "high" for s in by_step.values()])})
    if "step_log" in df.columns:
        df = pd.concat([df, df["step_log"].apply(step_stats)], axis=1)
 
    #scheme categories
    df["scheme"] = pd.Categorical(df["scheme"], categories=[s for s in SCHEMES if s in set(df["scheme"])], ordered=True)
    df["outcome"] = df["termination_reason"].map(outcome_group)
    return df

#helper function to get the schemes in the dataframe
def schemes_in(df) -> list[str]:
    return [s for s in df["scheme"].cat.categories]

#helper function to get the strategies in the dataframe
def strategies_in(df) -> list[str]:
    return [s for s in STRATEGIES if s in set(df["strategy"])]

#Tables
#plot the coverage of the grid
def coverage(df: pd.DataFrame) -> None:
    print_section("Grid Coverage (episodes per cell)")
    tbl = df.pivot_table(index="strategy", columns="scheme", values="task_id", aggfunc="count", observed=False)
    print(tbl.to_string())
    n_tasks = df["task_id"].nunique()
    print(f"\n {n_tasks} distinct tasks, budget {BUDGET:,}")

#success, cost and efficiency for every strategy x scheme cell
def grid_table(df: pd.DataFrame) -> pd.DataFrame:
    print_section("Success Rate and Token Efficiency (32k budget)")
    rows = []
    for (st, sc), g in df.groupby(["strategy", "scheme"], observed=True):
        k, n = int(g["success"].sum()), len(g)
        lo, hi = wilson(k, n)
        tok = g["total_tokens"]
        rows.append({
            "strategy": st, "scheme": sc, "n": n, "successes": k,
            "SR": k / n, "SR_95%CI": f"{lo:.2f}-{hi:.2f}",
            "mean_tokens": tok.mean(), "median_tokens": tok.median(),
            "tokens_per_success": tok.sum() / k if k else np.nan,
            "SR_per_1k_tokens": (k / n) / tok.mean() * 1000 if tok.mean() else np.nan,
            "mean_steps": g["steps"].mean(),
            "steps_trimmed": g["trim_share"].mean() if "trim_share" in g else np.nan,
            "steps_high_effort": g["high_share"].mean() if "high_share" in g else np.nan,
        })
    tbl = pd.DataFrame(rows).set_index(["strategy", "scheme"])
    with pd.option_context("display.float_format", lambda x: f"{x:,.3f}" if abs(x) < 10 else f"{x:,.0f}", "display.width", 220):
        print(tbl.to_string())
    return tbl

#paired comparison within each strategy
#paired on task: the same task under two schemes, within one strategy
def paired_scheme_tests(df: pd.DataFrame) -> None:
    print_section("Scheme vs Scheme within each Strategy (paired on task)")
    schemes = schemes_in(df)
    pairs = list(itertools.combinations(schemes, 2))
    print(f"success: exact McNemar. tokens: Wilcoxon signed-rank on the per-task difference.")
    for st in strategies_in(df):
        sub = df[df["strategy"] == st]
        rows = []
        for x, y in pairs:
            a = sub[sub["scheme"] == x].set_index("task_id")
            b = sub[sub["scheme"] == y].set_index("task_id")
            common = a.index.intersection(b.index)
            if not len(common):
                continue
            sa, sb = a.loc[common, "success"].astype(bool), b.loc[common, "success"].astype(bool)
            x_only, y_only = int((sa & ~sb).sum()), int((~sa & sb).sum())
            diff = b.loc[common, "total_tokens"].astype(float) - a.loc[common, "total_tokens"].astype(float)
            p_tok = np.nan
            if wilcoxon and (diff != 0).any():
                p_tok = wilcoxon(diff).pvalue
            rows.append({
                "comparison": f"{SCHEME_LABELS[y]} vs {SCHEME_LABELS[x]}",
                "tasks": len(common),
                f"SR_first": round(sb.mean(), 3), f"SR_second": round(sa.mean(), 3),
                "first_only": y_only, "second_only": x_only,
                "p_success": f"{mcnemar_exact(x_only, y_only):.4f}",
                "median_token_diff": f"{diff.median():+,.0f}",
                "p_tokens": f"{p_tok:.4f}" if not np.isnan(p_tok) else "n/a",
            })
        print(f" Strategy {st}")
        print(pd.DataFrame(rows).set_index("comparison").to_string())
        print()


#the same scheme, compared across strategies (A vs B vs C under identical allocation)
def paired_strategy_tests(df: pd.DataFrame) -> None:
    print_section("Strategy vs Strategy under the Same Scheme (paired on task)")
    pairs = [(x, y) for x, y in [("A", "B"), ("A", "C"), ("B", "C")] if x in strategies_in(df) and y in strategies_in(df)]
    rows = []
    for sc in schemes_in(df):
        sub = df[df["scheme"] == sc]
        for x, y in pairs:
            a = sub[sub["strategy"] == x].set_index("task_id")["success"]
            b = sub[sub["strategy"] == y].set_index("task_id")["success"]
            common = a.index.intersection(b.index)
            if not len(common):
                continue
            #get the success rates
            sa, sb = a.loc[common].astype(bool), b.loc[common].astype(bool)
            x_only, y_only = int((sa & ~sb).sum()), int((~sa & sb).sum())
            rows.append({"scheme": SCHEME_LABELS[sc], "pair": f"{x} vs {y}", "tasks": len(common),
                         f"SR_first": round(sa.mean(), 3), f"SR_second": round(sb.mean(), 3),
                         "first_only": x_only, "second_only": y_only,
                         "p": f"{mcnemar_exact(x_only, y_only):.4f}"})
    print(pd.DataFrame(rows).set_index(["scheme", "pair"]).to_string())


#paired bootstrap over tasks: each scheme's change against a reference scheme, per strategy
def scheme_effects(df: pd.DataFrame, reference: str, reps: int = 2000, seed: int = 42) -> pd.DataFrame | None:
    if reference not in schemes_in(df):
        return None
    print_section(f"Effect of each Scheme relative to {SCHEME_LABELS[reference]} (paired bootstrap over tasks)")
    rng = np.random.default_rng(seed)
    rows = []
    for st in strategies_in(df):
        sub = df[df["strategy"] == st]
        ref = sub[sub["scheme"] == reference].set_index("task_id")
        for sc in schemes_in(df):
            if sc == reference:
                continue
            cur = sub[sub["scheme"] == sc].set_index("task_id")
            common = ref.index.intersection(cur.index)
            if not len(common):
                continue
            s0 = ref.loc[common, "success"].astype(float).to_numpy()
            s1 = cur.loc[common, "success"].astype(float).to_numpy()
            t0 = ref.loc[common, "total_tokens"].astype(float).to_numpy()
            t1 = cur.loc[common, "total_tokens"].astype(float).to_numpy()
            idx = rng.integers(0, len(common), size=(reps, len(common)))
            d_sr = (s1[idx].mean(1) - s0[idx].mean(1)) * 100
            d_tok = (t1[idx].sum(1) / np.maximum(t0[idx].sum(1), 1) - 1) * 100
            rows.append({
                "strategy": st, "scheme": sc, "tasks": len(common),
                "dSR_pp": (s1.mean() - s0.mean()) * 100,
                "dSR_lo": np.percentile(d_sr, 2.5), "dSR_hi": np.percentile(d_sr, 97.5),
                "dTokens_%": (t1.sum() / t0.sum() - 1) * 100 if t0.sum() else np.nan,
                "dTok_lo": np.percentile(d_tok, 2.5), "dTok_hi": np.percentile(d_tok, 97.5),
            })
    tbl = pd.DataFrame(rows)
    show = tbl.copy()
    show["change in SR (pp)"] = show.apply(lambda r: f"{r['dSR_pp']:+.1f}  [{r['dSR_lo']:+.1f}, {r['dSR_hi']:+.1f}]", axis=1)
    show["change in tokens (%)"] = show.apply(lambda r: f"{r['dTokens_%']:+.1f}  [{r['dTok_lo']:+.1f}, {r['dTok_hi']:+.1f}]", axis=1)
    show["scheme"] = show["scheme"].map(SCHEME_LABELS)
    print(show.set_index(["strategy", "scheme"])[["tasks", "change in SR (pp)", "change in tokens (%)"]].to_string())
    print("\n [95% interval]. An interval that excludes 0 is a clear effect.")
    return tbl


#does the best scheme depend on the strategy? ranking of schemes within each strategy
def scheme_ranking(tbl: pd.DataFrame) -> None:
    print_section("Which Scheme Suits which Strategy")
    for st in tbl.index.get_level_values("strategy").unique():
        t = tbl.loc[st]
        best_sr = t["SR"].idxmax()
        best_eff = t["tokens_per_success"].idxmin()
        print(f" Strategy {st}: highest success = {SCHEME_LABELS[best_sr]} ({t.loc[best_sr, 'SR']:.1%}), "
              f"cheapest per success = {SCHEME_LABELS[best_eff]} ({t.loc[best_eff, 'tokens_per_success']:,.0f} tokens)")


def outcome_mix_table(df: pd.DataFrame) -> None:
    print_section("How Episodes End, per Strategy and Scheme")
    tbl = pd.crosstab([df["strategy"], df["scheme"]], df["outcome"], normalize="index")
    tbl = tbl[[g for g in OUTCOME_ORDER if g in tbl.columns]]
    with pd.option_context("display.float_format", lambda x: f"{x:.0%}", "display.width", 200):
        print(tbl.to_string())


#secondary: ~33 tasks per tier, so descriptive only
def tier_table(df: pd.DataFrame) -> None:
    if "difficulty_tier" not in df.columns or df["difficulty_tier"].isna().all():
        return
    print_section("Success Rate by Difficulty Tier (descriptive, ~33 tasks per tier)")
    piv = df.pivot_table(values="success", index=["strategy", "difficulty_tier"], columns="scheme", aggfunc="mean", observed=False)
    piv = piv.reindex([(s, t) for s in strategies_in(df) for t in TIER_ORDER])
    piv.columns = [SCHEME_LABELS[c] for c in piv.columns]
    with pd.option_context("display.float_format", lambda x: f"{x:.2f}"):
        print(piv.to_string())

#Plots
#helper function to save the plot
def _finish(fig, ax, out_dir: Path, name: str, legend: bool = True) -> None:
    if legend:
        ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved: {path}")
    plt.close(fig)

#grouped bars: x = scheme, one bar per strategy (strategy keeps its colour everywhere)
def _grouped_bars(df, out_dir, name, title, ylabel, value, err=None, fmt=None):
    schemes, strats = schemes_in(df), strategies_in(df)
    x = np.arange(len(schemes))
    width = 0.8 / len(strats)
    fig, ax = plt.subplots()
    for i, st in enumerate(strats):
        vals, lo_err, hi_err = [], [], []
        for sc in schemes:
            g = df[(df["strategy"] == st) & (df["scheme"] == sc)]
            v = value(g) if len(g) else np.nan
            vals.append(v)
            if err:
                lo, hi = err(g) if len(g) else (np.nan, np.nan)
                lo_err.append(max(0, v - lo) if len(g) else 0)
                hi_err.append(max(0, hi - v) if len(g) else 0)
        kw = dict(yerr=[lo_err, hi_err], capsize=3, error_kw={"elinewidth": 1, "ecolor": "#52514e"}) if err else {}
        bars = ax.bar(x + (i - (len(strats) - 1) / 2) * width, vals, width * 0.92, color=COLOURS[st], label=NAMES[st], **kw)
        if fmt:
            for b, v in zip(bars, vals):
                if not np.isnan(v):
                    ax.annotate(fmt(v), (b.get_x() + b.get_width() / 2, 0), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=8, color="white")
    ax.set_xticks(x)
    ax.set_xticklabels([SCHEME_LABELS[s] for s in schemes])
    ax.set_xlabel("Spending Scheme (32k budget)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _finish(fig, ax, out_dir, name)


def plot_success(df, out_dir):
    _grouped_bars(df, out_dir, "success_by_scheme.png", "Success Rate by Spending Scheme and Strategy", "Success Rate (95% interval)",
                  lambda g: g["success"].mean(), err=lambda g: wilson(int(g["success"].sum()), len(g)))

def plot_tokens(df, out_dir):
    _grouped_bars(df, out_dir, "rq3_tokens_by_scheme.png", "Mean Tokens Spent by Spending Scheme and Strategy",
                  "Mean Tokens per Episode", lambda g: g["total_tokens"].mean())

def plot_tokens_per_success(df, out_dir):
    _grouped_bars(df, out_dir, "rq3_tokens_per_success.png", "Token Cost per Success by Spending Scheme and Strategy",
                  "Tokens per Successful Episode (lower is better)", lambda g: g["total_tokens"].sum() / g["success"].sum() if g["success"].sum() else np.nan)


#marker shape = scheme, colour = strategy (the colour a strategy has in every figure)
SCHEME_MARKERS = {"pay_as_you_go": "o", "even": "s", "front_loaded": "^", "reactive": "D"}

#helper function to get the points for a cell
def _cell_points(df, st):
    pts = []
    for sc in schemes_in(df):
        g = df[(df["strategy"] == st) & (df["scheme"] == sc)]
        if not g.empty:
            pts.append((sc, g["total_tokens"].mean(), g["success"].mean()))
    return pts

#plot the cost vs success for each strategy and scheme
def plot_cost_performance(df, out_dir):
    #get the mean tokens and success rates for each strategy and scheme
    xs = df.groupby(["strategy", "scheme"], observed=True)["total_tokens"].mean()
    ys = df.groupby(["strategy", "scheme"], observed=True)["success"].mean()
    xlim = (xs.min() * 0.9, xs.max() * 1.05)
    ylim = (0, ys.max() * 1.25)

    fig, ax = plt.subplots()
    for st in strategies_in(df):
        for sc, xt, yv in _cell_points(df, st):
            ax.plot(xt, yv, SCHEME_MARKERS[sc], color=COLOURS[st], markersize=10, markeredgecolor="white", markeredgewidth=1.5)
    handles = ([Line2D([], [], marker="o", linestyle="none", color=COLOURS[st], markersize=9, label=NAMES[st])
                for st in strategies_in(df)] +
               [Line2D([], [], marker=SCHEME_MARKERS[sc], linestyle="none", color="#52514e", markersize=8, label=SCHEME_LABELS[sc]) for sc in schemes_in(df)])
    ax.legend(handles=handles, ncol=2, fontsize=9, loc="lower left")
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_xlabel("Mean Tokens Spent per Episode")
    ax.set_ylabel("Success Rate")
    ax.set_title("Cost vs Success for each Spending Scheme (32k budget)")
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
    _finish(fig, ax, out_dir, "cost_vs_success.png", legend=False)
 
    #one labelled version per strategy: zoomed to its own token range (same y-scale), 
    #labels alternate above/below so neighbouring points stay readable
    for st in strategies_in(df):
        #get the points for the strategy
        pts = sorted(_cell_points(df, st), key=lambda p: p[1])
        #get the min and max tokens
        lo_x, hi_x = min(p[1] for p in pts), max(p[1] for p in pts)
        #get the padding
        pad = max((hi_x - lo_x) * 0.25, 500)
        #create the figure and plot the points
        fig, ax = plt.subplots()
        for i, (sc, xt, yv) in enumerate(pts):
            ax.plot(xt, yv, SCHEME_MARKERS[sc], color=COLOURS[st], markersize=11, markeredgecolor="white", markeredgewidth=1.5)
            ax.annotate(f"{SCHEME_LABELS[sc]}  ({yv:.0%}, {xt / 1000:.1f}k)", (xt, yv), textcoords="offset points", xytext=(0, 12 if i % 2 == 0 else -18), ha="center", fontsize=9, color="#52514e")
        ax.set_xlim(lo_x - pad, hi_x + pad)
        ax.set_ylim(*ylim)
        ax.set_xlabel("Mean Tokens Spent per Episode")
        ax.set_ylabel("Success Rate")
        ax.set_title(f"Strategy {st}: Cost vs Success for each Spending Scheme (32k budget)")
        #set the x-axis major formatter
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v / 1000:g}k"))
        #save the figure
        _finish(fig, ax, out_dir, f"cost_vs_success_{st.lower()}.png", legend=False)

#plot the effects of each scheme relative to a reference scheme
def plot_effects(eff: pd.DataFrame | None, reference: str, out_dir: Path):
    if eff is None or eff.empty:
        return
    #plot the effects of each scheme relative to a reference scheme
    #dSr = change in success rate in percentage points
    #dTokens_% = change in tokens spent in percentage
    #lo = lower bound of the 95% bootstrap interval
    #hi = upper bound of the 95% bootstrap interval
    #label = label for the x-axis
    for col, lo, hi, label, name in [
            ("dSR_pp", "dSR_lo", "dSR_hi", "Change in Success Rate (percentage points)", "effect_success.png"),
            ("dTokens_%", "dTok_lo", "dTok_hi", "Change in Tokens Spent (%)", "effect_tokens.png")]:
        fig, ax = plt.subplots(figsize=(10, 1.6 + 0.45 * len(eff)))
        ys, labels = [], []
        y = 0
        for st in [s for s in STRATEGIES if s in set(eff["strategy"])]:
            for _, r in eff[eff["strategy"] == st].iterrows():
                ax.errorbar(r[col], y, xerr=[[max(0, r[col] - r[lo])], [max(0, r[hi] - r[col])]], fmt=MARKERS[st][0], color=COLOURS[st], markersize=8, capsize=3, elinewidth=1.5)
                labels.append(f"{st}  {SCHEME_LABELS[r['scheme']]}")
                ys.append(y)
                y += 1
            y += 0.6
        ax.axvline(0, color="#9E9E9E", linewidth=1, linestyle=":")
        ax.set_yticks(ys)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(f"{label} vs {SCHEME_LABELS[reference]}, 95% bootstrap interval")
        ax.set_title(f"Effect of each Scheme relative to {SCHEME_LABELS[reference]}")
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        path = out_dir / name
        fig.savefig(path, dpi=150)
        print(f"saved: {path}")
        plt.close(fig)

#plot the outcome mix for each strategy and scheme
def plot_outcome_mix(df, out_dir):
    labels, shares = [], []
    for st in strategies_in(df):
        for sc in schemes_in(df):
            g = df[(df["strategy"] == st) & (df["scheme"] == sc)]
            if g.empty:
                continue
            #get the counts of the outcomes
            c = g["outcome"].value_counts()
            #add the labels and shares
            labels.append(f"{st}  {SCHEME_LABELS[sc]}")
            shares.append(c / c.sum())
    #get the groups of outcomes
    groups = [g for g in OUTCOME_ORDER if any(s.get(g, 0) for s in shares)]
    #get the number of schemes
    per = len(schemes_in(df))
    #get the y positions for the bars
    y = np.array([i + (i // per) * 0.6 for i in range(len(labels))])
    fig, ax = plt.subplots(figsize=(10, 1.6 + 0.45 * len(labels)))
    left = np.zeros(len(labels))
    for g in groups:
        #get the values for the group
        vals = np.array([s.get(g, 0) for s in shares])
        #plot the bars
        ax.barh(y, vals, left=left, height=0.85, color=OUTCOME_COLOURS[g], edgecolor="white", linewidth=2, label=g)
        #add the labels
        for yi, v, l in zip(y, vals, left):
            #only add the label if the value is greater than 7%
            if v >= 0.07:
                ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of Episodes")
    ax.set_title("How Episodes End under each Spending Scheme")
    ax.legend(ncol=len(groups), loc="upper center", bbox_to_anchor=(0.5, -0.06 - 1.0 / len(labels)), frameon=False, fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    path = out_dir / "outcome_mix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved: {path}")
    plt.close(fig)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schemes", default="../data/processed/schemes_output/schemes.jsonl")
    ap.add_argument("--main-a", default=None)
    ap.add_argument("--main-b", default=None)
    ap.add_argument("--main-c", default=None)
    ap.add_argument("--tasks", default="../data/processed/task_list/read_only_tasks.jsonl")
    ap.add_argument("--reference", default=None)
    ap.add_argument("--out", default="../data/processed/diagrams/schemes_output")
    args = ap.parse_args()
 
    df = load_grid(args)
    reference = args.reference or ("pay_as_you_go" if "pay_as_you_go" in schemes_in(df) else "even")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
 
    coverage(df)
    tbl = grid_table(df)
    paired_scheme_tests(df)
    eff = scheme_effects(df, reference)
    paired_strategy_tests(df)
    scheme_ranking(tbl)
    outcome_mix_table(df)
    tier_table(df)
 
    print_section("Plots")
    plot_success(df, out_dir)
    plot_tokens(df, out_dir)
    plot_tokens_per_success(df, out_dir)
    plot_cost_performance(df, out_dir)
    plot_effects(eff, reference, out_dir)
    plot_outcome_mix(df, out_dir)
    tbl.to_csv(out_dir / "grid.csv")
    print(f"saved: {out_dir / 'grid.csv'}")
 
 
if __name__ == "__main__":
    main()