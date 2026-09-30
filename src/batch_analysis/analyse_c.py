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