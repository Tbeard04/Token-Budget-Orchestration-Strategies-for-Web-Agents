from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from batch_analysis.shared import (load, join_tiers, print_section, run_shared_analysis, wilson,COLOURS, ROUTER_COLOUR, TIER_COLOURS, TIER_ORDER)

#pull the router block's fields up into ordinary columns
def expand_router(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["router_stop"] = df["termination_reason"] == "router_stop"
    if "router" not in df.columns:
        print("[analyse_c] no 'router' block in this file - routing sections skipped")
        return df
    r = df["router"].apply(lambda x: x if isinstance(x, dict) else {})
    df["stop_step"] = r.apply(lambda x: x.get("stop_step"))
    df["p_cycle"] = r.apply(lambda x: x.get("p_cycle"))
    df["p_stop_step0"] = r.apply(lambda x: (x.get("decisions") or [{}])[0].get("p_stop"))
    df["stop_threshold"] = r.apply(lambda x: x.get("stop_threshold"))
    if "mode_chosen" not in df.columns:
        df["mode_chosen"] = r.apply(lambda x: x.get("mode"))
    return df

#Table to print whether C's router drop come from stopping (a choice) or from failing attempts
def per_tier_router(df: pd.DataFrame) -> None:
    if "difficulty_tier" not in df.columns:
        return
    print_section("Router by Difficulty Tier (RQ2)")
    rows = []
    for tier in TIER_ORDER:
        g = df[df["difficulty_tier"] == tier]
        if g.empty:
            continue
        att = g[~g["router_stop"]]
        rows.append({
            "tier": tier,
            "episodes": len(g),
            "stop_rate": g["router_stop"].mean(),
            "SR_all": g["success"].mean(),
            "SR_attempted": att["success"].mean() if len(att) else float("nan"),
            "gap": (att["success"].mean() if len(att) else 0) - g["success"].mean(),
        })
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(pd.DataFrame(rows).set_index("tier").to_string())

#table for router by budget: stop rate, and success over all vs over attempted episodes
def per_budget_router(df: pd.DataFrame) -> pd.DataFrame:
    print_section("Router by Budget: Stops and the Two Success Rates")
    rows = []
    for budget, g in df.groupby("budget_level"):
        att = g[~g["router_stop"]]
        rows.append({
            "budget": f"{budget // 1000}k",
            "episodes": len(g),
            "router_stops": int(g["router_stop"].sum()),
            "stop_rate": g["router_stop"].mean(),
            "SR_all": g["success"].mean(),
            "attempted": len(att),
            "SR_attempted": att["success"].mean() if len(att) else float("nan"),
            "mean_tokens": g["total_tokens"].mean(),
            "cycle_share": (g["mode_chosen"] == "cycle").mean() if "mode_chosen" in g else float("nan"),
        })
    tbl = pd.DataFrame(rows).set_index("budget")
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}", "display.width", 200):
        print(tbl.to_string())
    return tbl

def _finish(fig, ax, out_dir: Path, name: str, legend: bool = True) -> None:
    if legend:
        ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")
    plt.close(fig)
 
 
def plot_two_success_rates(df: pd.DataFrame, out_dir: Path) -> None:
    budgets = sorted(df["budget_level"].unique())
    all_sr, att_sr, all_ci = [], [], []
    for b in budgets:
        g = df[df["budget_level"] == b]
        att = g[~g["router_stop"]]
        all_sr.append(g["success"].mean())
        att_sr.append(att["success"].mean() if len(att) else np.nan)
        all_ci.append(wilson(int(g["success"].sum()), len(g)))
 
    fig, ax = plt.subplots()
    lo, hi = zip(*all_ci)
    ax.fill_between(budgets, lo, hi, color=COLOURS["C"], alpha=0.12, linewidth=0)
    ax.plot(budgets, all_sr, "^-", color=COLOURS["C"], linewidth=2, markersize=8, label="All episodes (router stops count as failures)")
    ax.plot(budgets, att_sr, "o--", color=ROUTER_COLOUR, linewidth=2, markersize=8, label="Attempted episodes only (router stops removed)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels([f"{b // 1000}k" for b in budgets])
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Success Rate")
    ax.set_title("Strategy C: Success Rate, All vs Attempted Episodes")
    _finish(fig, ax, out_dir, "success_all_vs_attempted_c.png")
 
 
def plot_stop_rate_by_budget(df: pd.DataFrame, out_dir: Path) -> None:
    g = df.groupby("budget_level")["router_stop"].agg(["sum", "count"])
    rate = g["sum"] / g["count"]
    lo, hi = zip(*[wilson(int(k), int(n)) for k, n in zip(g["sum"], g["count"])])
 
    fig, ax = plt.subplots()
    x = np.arange(len(g))
    ax.bar(x, rate, 0.6, color=ROUTER_COLOUR,
           yerr=[rate - np.array(lo), np.array(hi) - rate], capsize=3,
           error_kw={"elinewidth": 1, "ecolor": "#52514e"})
    #episode count above each bar, offset right of the error bar
    for xi, (r, k) in enumerate(zip(rate, g["sum"])):
        ax.annotate(f"{int(k)} stopped", (xi, r), textcoords="offset points", xytext=(12, 3), fontsize=9, color="#52514e")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b // 1000}k" for b in g.index])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Episodes Stopped by the Router")
    ax.set_title("Strategy C: Router Stop Rate by Budget")
    _finish(fig, ax, out_dir, "router_stop_rate_c.png", legend=False)


def plot_stop_rate_by_tier(df: pd.DataFrame, out_dir: Path) -> None:
    if "difficulty_tier" not in df.columns:
        return
    budgets = sorted(df["budget_level"].unique())
    x = np.arange(len(budgets))
    width = 0.8 / len(TIER_ORDER)
 
    fig, ax = plt.subplots()
    for i, tier in enumerate(TIER_ORDER):
        sub = df[df["difficulty_tier"] == tier]
        rates = [sub[sub["budget_level"] == b]["router_stop"].mean() if len(sub[sub["budget_level"] == b]) else 0 for b in budgets]
        ax.bar(x + (i - 1) * width, rates, width * 0.92, color=TIER_COLOURS[tier], label=tier)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{b // 1000}k" for b in budgets])
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Episodes Stopped by the Router")
    ax.set_title("Strategy C: Router Stop Rate by Difficulty Tier")
    ax.legend(title="Tier")
    _finish(fig, ax, out_dir, "router_stop_by_tier_c.png", legend=False)


def plot_stop_step(df: pd.DataFrame, out_dir: Path) -> None:
    if "stop_step" not in df.columns or not df["router_stop"].any():
        return
    steps = df.loc[df["router_stop"], "stop_step"].dropna().astype(int)
    counts = steps.value_counts().sort_index()

    fig, ax = plt.subplots()
    ax.bar(counts.index, counts.values, 0.6, color=ROUTER_COLOUR)
    ax.set_xticks(counts.index)
    ax.set_xlabel("Step at Which the Router Stopped (0 = before any action)")
    ax.set_ylabel("Episodes")
    ax.set_title("Strategy C: When the Router Stops")
    _finish(fig, ax, out_dir, "router_stop_step_c.png", legend=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--tiers", default=None)
    ap.add_argument("--all-rows", action="store_true")
    ap.add_argument("--verbose-tasks", action="store_true")
    ap.add_argument("--out", default="../data/processed/diagrams/6_budgets_ALL_tasks_decontaminted_batches/strategy_c_batch")
    args = ap.parse_args()

    df = load(args.file, clean=not args.all_rows)
    if args.tiers:
        df = join_tiers(df, args.tiers)
    df = expand_router(df)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_shared_analysis(df, out_dir, verbose_tasks=args.verbose_tasks)

    per_budget_router(df)
    per_tier_router(df)

    print_section("Strategy C Plots")
    plot_two_success_rates(df, out_dir)
    plot_stop_rate_by_budget(df, out_dir)
    plot_stop_rate_by_tier(df, out_dir)
    plot_stop_step(df, out_dir)


if __name__ == "__main__":
    main()