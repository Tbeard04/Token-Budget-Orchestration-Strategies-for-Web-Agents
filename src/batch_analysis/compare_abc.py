"""
analysis/compare_ab.py - Strategy A vs Strategy B vs Strategy C comparison
"""
from __future__ import annotations

import argparse
import statistics
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from batch_analysis.shared import (load, join_tiers, print_section, wilson, mcnemar_exact, outcome_group,COLOURS, MARKERS, NAMES, TIER_ORDER, OUTCOME_ORDER, OUTCOME_COLOURS)

#pairs of strategies to compare
PAIRS = [("A", "B"), ("A", "C"), ("B", "C")]

#"A vs B" or "A vs B vs C" depending on what was loaded
def vs(d: dict) -> str:
    return " vs ".join(d)


#the strategy pairs that were loaded
def present_pairs(d: dict) -> list[tuple[str, str]]:
    return [(x, y) for x, y in PAIRS if x in d and y in d]

#function to label the budgets
def budget_labels(budgets) -> list[str]:
    return [f"{b // 1000}k" for b in budgets]

#table for success rate by budget
def comparison_table(d: dict[str, pd.DataFrame]) -> pd.DataFrame:
    print_section(f"{vs(d)}: Success Rate by Budget")

    budgets = sorted(set().union(*[set(df["budget_level"]) for df in d.values()]))
    rows = []
    #loop through each budget level
    for budget in budgets:
        row = {"budget": budget}
        #loop through each strategy
        for s, df in d.items():
            #filter the dataframe to only include episodes in the current budget level
            g = df[df["budget_level"] == budget]
            #append the number of episodes to the row
            row[f"{s}_n"] = len(g)
            row[f"{s}_SR"] = g["success"].mean() if len(g) else float("nan")
            row[f"{s}_med_tok"] = g["total_tokens"].median() if len(g) else float("nan")
        rows.append(row)
    tbl = pd.DataFrame(rows).set_index("budget")
    #success-rate gap for each pair, second minus first (positive = second is better)
    for x, y in present_pairs(d):
        tbl[f"{y}-{x}_SR"] = tbl[f"{y}_SR"] - tbl[f"{x}_SR"]

    with pd.option_context("display.float_format", lambda x: f"{x:.3f}", "display.width", 200):
        print(tbl.to_string())
    return tbl

#function to calculate the equivalent budget
def equivalent_budget(d: dict[str, pd.DataFrame]) -> None:
    #if A is not in the dictionary, return
    if "A" not in d:
        return
    print_section(f"{vs(d)}: Equivalent Budget (relative to A)")

    #group the dataframe by budget level and calculate the success rate
    a_sr = d["A"].groupby("budget_level")["success"].mean()
    #loop through each strategy
    for s, df in d.items():
        if s == "A":
            continue
        print(f"\n Strategy {s}")
        for s_budget, s_val in df.groupby("budget_level")["success"].mean().items():
            #calculate the closest budget level
            closest = (a_sr - s_val).abs().idxmin()
            #calculate the ratio of the current budget level to the closest budget level
            ratio = s_budget / closest if closest else float("inf")
            print(f" {s} @ {s_budget//1000:>2}k ({s_val:>5.1%})  "
                  f"~=  A @ {closest//1000:>2}k ({a_sr[closest]:>5.1%})   "
                  f"[{s} needs {ratio:.1f}x the budget]")

#table for cost ratio on shared successes
def cost_ratio(d: dict[str, pd.DataFrame]) -> None:
    if "A" not in d:
        return
    print_section(f"{vs(d)}: Cost Ratio on Shared Successes (relative to A)")
    #get the dataframe for A
    a = d["A"]
    a_solved = set(a.loc[a["success"] == True, "task_id"])
    #loop through each strategy
    for s, df in d.items():
        #if the strategy is A, skip to the next strategy
        if s == "A":
            continue
        #get the tasks solved by the current strategy
        s_solved = set(df.loc[df["success"] == True, "task_id"])
        ratios = []
        for tid in a_solved & s_solved:
            #get the cheapest success of each strategy on this task
            a_cost = a.loc[(a["task_id"] == tid) & (a["success"] == True), "total_tokens"].min()
            #get the cheapest success of the current strategy on this task
            s_cost = df.loc[(df["task_id"] == tid) & (df["success"] == True), "total_tokens"].min()
            #if the cost is not None and greater than 0, append the ratio to the ratios list
            if a_cost and a_cost > 0:
                ratios.append(s_cost / a_cost)
        print(f"\n Strategy {s}")
        #if there are no shared successes, print a message and skip to the next strategy
        if not ratios:
            print("no shared successes to compare")
            continue
        print(f"tasks solved by both: {len(ratios)}")
        print(f"median {s}/A cost ratio: {statistics.median(ratios):.2f}x")
        print(f"mean {s}/A cost ratio: {statistics.mean(ratios):.2f}x")
        print(f"range: {min(ratios):.2f}x - {max(ratios):.2f}x")


#same task, same budget, both strategies kept after decontamination ---> one pair
#exact McNemar on the pairs where only one of the two succeeded
#McNemar, Q. (1947) "Note on the sampling error of the difference between correlated proportions or percentages"
#AI-Generated
def paired_tests(d: dict[str, pd.DataFrame]) -> None:
    print_section(f"{vs(d)}: Paired Comparison (same task, same budget), exact McNemar")

    key = ["task_id", "budget_level"]
    for x, y in present_pairs(d):
        m = d[x][key + ["success"]].merge(d[y][key + ["success"]], on=key, suffixes=(f"_{x}", f"_{y}"))
        rows = []
        groups = [(b, g) for b, g in m.groupby("budget_level")] + [("all", m)]
        for budget, g in groups:
            sx = g[f"success_{x}"].astype(bool)
            sy = g[f"success_{y}"].astype(bool)
            x_only = int((sx & ~sy).sum())
            y_only = int((~sx & sy).sum())
            rows.append({
                "budget": budget if budget == "all" else f"{budget // 1000}k",
                "pairs": len(g),
                f"{x}_SR": round(sx.mean(), 3),
                f"{y}_SR": round(sy.mean(), 3),
                "both": int((sx & sy).sum()),
                f"{x}_only": x_only,
                f"{y}_only": y_only,
                "p": round(mcnemar_exact(x_only, y_only), 4),
            })
        print(f" {x} vs {y}")
        print(pd.DataFrame(rows).set_index("budget").to_string())
        print()
#AI-Generated

#router vs fixed policies comparison
def router_vs_fixed_policies(d: dict[str, pd.DataFrame]) -> None:
    if not all(s in d for s in "ABC"):
        return
    print_section("Strategy C vs Fixed Policies (same task, same budget)")
    #get the key columns
    key = ["task_id", "budget_level"]
    cols = key + ["success", "total_tokens"]
    m = (d["A"][cols].merge(d["B"][cols], on=key, suffixes=("_A", "_B")).merge(d["C"][cols].rename(columns={"success": "success_C", "total_tokens": "total_tokens_C"}), on=key))
    #get the success and total tokens for each strategy
    sa, sb, sc = (m[f"success_{s}"].astype(bool) for s in "ABC")
    ta, tb, tc = (m[f"total_tokens_{s}"] for s in "ABC")
    #oracle: knows the outcomes.
    #get the oracle success and tokens
    #Takes the cheaper arm that succeeded, or stops at zero cost when neither did
    oracle_success = sa | sb
    oracle_tokens = np.where(sa & sb, np.minimum(ta, tb), np.where(sa, ta, np.where(sb, tb, 0)))
    #get the base success and tokens
    base_s, base_t = int(sa.sum()), int(ta.sum())
    #initialize the rows list
    rows = []
    #loop through the policies
    for name, s, t in [("Always A", sa, ta), ("Always B", sb, tb), ("Strategy C (live)", sc, tc), ("Oracle", oracle_success, oracle_tokens)]:
        s_n, t_n = int(s.sum()), int(np.sum(t))
        #append the data for the current policy to the rows list
        rows.append({
            "policy": name,
            "successes": s_n,
            "kept_vs_A": f"{s_n / base_s:.0%}" if base_s else "n/a",
            "tokens": f"{t_n:,}",
            "saved_vs_A": f"{1 - t_n / base_t:.0%}" if base_t else "n/a",
        })
    print(f" {len(m)} cells shared by A, B and C\n")
    print(pd.DataFrame(rows).set_index("policy").to_string())
    #get the number of episodes that stopped the router
    stopped = (d["C"]["termination_reason"] == "router_stop").sum()
    print(f"\n C router stops: {stopped} of {len(d['C'])} episodes")
    #if the mode chosen column is in the dataframe, get the value counts
    if "mode_chosen" in d["C"].columns:
        #get the value counts of the mode chosen column
        modes = d["C"]["mode_chosen"].value_counts()
        print(" C mode choices: " + ", ".join(f"{k} {v}" for k, v in modes.items()) + "   (execute = A's way, cycle = B's way)")

#differences in task-level performance
def task_level_comparison(d: dict[str, pd.DataFrame]) -> None:
    print_section(f"{vs(d)}: Task-Level Comparison (solved at any budget)")
    #get the names of the strategies
    names = list(d)
    #get the tasks solved by each strategy
    solved = {s: set(df.loc[df["success"] == True, "task_id"]) for s, df in d.items()}
    #get the all tasks
    all_tasks = set().union(*[set(df["task_id"]) for df in d.values()])

    #every solved / not-solved pattern across the loaded strategies
    for pattern in itertools.product([True, False], repeat=len(names)):
        #get the tasks that match the pattern
        tasks = {t for t in all_tasks if all((t in solved[s]) == want for s, want in zip(names, pattern))}
        #get the winners
        winners = [s for s, want in zip(names, pattern) if want]
        #get the label for the winners
        label = ("neither" if len(names) == 2 else "none") if not winners else \
            (" + ".join(winners) + (" only" if len(winners) < len(names) else " (all)"))
        print(f"solved by {label:14s} {len(tasks):>4}")
    #if A and B are in the dictionary, get the tasks solved by B but not A
    if "A" in d and "B" in d:
        #get the dataframes for A and B
        a, b = d["A"], d["B"]
        #get the tasks solved by B but not A
        b_only = solved["B"] - solved["A"]
        #if there are tasks solved by B but not A, print the tasks
        if b_only:
            print(f"\n Tasks B solved that A could not:")
            with_rev = 0
            for tid in sorted(b_only):
                ep = b[(b["task_id"] == tid) & (b["success"] == True)]
                ep = ep.loc[ep["budget_level"].idxmin()]
                revs = int(ep.get("critic_revisions", 0) or 0)
                with_rev += bool(revs)
                a_eps = a[a["task_id"] == tid]
                a_reason = a_eps.loc[a_eps["budget_level"].idxmax(), "termination_reason"] if len(a_eps) else "not run"
                print(f"task {tid:>4} @ {ep['budget_level']:>6}: {ep['steps']} steps, "
                      f"{revs} revisions   (A failed: {a_reason})")
            print(f"\n {with_rev}/{len(b_only)} involved a Critic revision")
            print(f"{len(b_only) - with_rev}/{len(b_only)} succeeded with the Planner alone")

    #did the Critic contribute to B's wins over A
    if "A" in d and "C" in d:
        c = d["C"]
        lost = solved["A"] - solved["C"]
        gained = solved["C"] - solved["A"]
        print(f"\n Tasks A solved that C never did: {len(lost)}")
        #if there are tasks solved by C but not A, print the tasks
        if lost:
            why = c[c["task_id"].isin(lost)]["termination_reason"].map(outcome_group).value_counts()
            print("C's episodes on those tasks ended as: " + ", ".join(f"{k} {v}" for k, v in why.items()))
        print(f"Tasks C solved that A never did: {len(gained)}"
              + (f"{sorted(gained)[:15]}" if gained else ""))

#helper function to check if the data has tiers
def _has_tiers(d: dict) -> bool:
    return all("difficulty_tier" in df.columns for df in d.values())


#success rate per tier (all budgets pooled) with 95% intervals, plus tier x budget per strategy
def difficulty_table(d: dict[str, pd.DataFrame]) -> None:
    if not _has_tiers(d):
        return
    print_section(f"{vs(d)}: Success Rate by Difficulty Tier")
    rows = []
    #loop through each tier
    for tier in TIER_ORDER:
        #initialize the row
        row = {"tier": tier}
        #loop through each strategy
        for s, df in d.items():
            #filter the dataframe to only include episodes in the current tier
            g = df[df["difficulty_tier"] == tier]["success"]
            #get the number of successes and the number of episodes
            k, n = int(g.sum()), len(g)
            #calculate the Wilson score interval
            lo, hi = wilson(k, n)
            #append the number of successes and the number of episodes to the row
            row[f"{s}_n"] = n
            #append the success rate to the row
            row[f"{s}_SR"] = f"{k / n:.1%}" if n else "n/a"
            #append the 95% confidence interval to the row
            row[f"{s}_95%CI"] = f"{lo:.1%}-{hi:.1%}" if n else ""
        #append the row to the rows list
        rows.append(row)
    print(pd.DataFrame(rows).set_index("tier").to_string())

    print("\n success rate by tier and budget:")
    piv = pd.concat({s: df.pivot_table(values="success", index="difficulty_tier", columns="budget_level", aggfunc="mean").reindex(TIER_ORDER) for s, df in d.items()}, names=["strategy", "tier"])
    piv.columns = budget_labels(piv.columns)
    with pd.option_context("display.float_format", lambda x: f"{x:.3f}"):
        print(piv.to_string())


#same task, same budget, both strategies kept: McNemar within each tier (budgets pooled)
#AI-Generated
def paired_tests_by_tier(d: dict[str, pd.DataFrame]) -> None:
    if not _has_tiers(d):
        return
    print_section(f"{vs(d)}: Paired Comparison within each Difficulty Tier, exact McNemar")
    key = ["task_id", "budget_level"]
    for x, y in present_pairs(d):
        m = d[x][key + ["success", "difficulty_tier"]].merge(
            d[y][key + ["success"]], on=key, suffixes=(f"_{x}", f"_{y}"))
        rows = []
        for tier in TIER_ORDER:
            g = m[m["difficulty_tier"] == tier]
            if g.empty:
                continue
            sx = g[f"success_{x}"].astype(bool)
            sy = g[f"success_{y}"].astype(bool)
            x_only, y_only = int((sx & ~sy).sum()), int((~sx & sy).sum())
            rows.append({"tier": tier, "pairs": len(g), f"{x}_SR": round(sx.mean(), 3),
                         f"{y}_SR": round(sy.mean(), 3), f"{x}_only": x_only, f"{y}_only": y_only, "p": f"{mcnemar_exact(x_only, y_only):.4f}"})
        print(f" {x} vs {y}")
        print(pd.DataFrame(rows).set_index("tier").to_string())
        print()
#AI-Generated


#difficulty sensitivity analysis
#how much each strategy loses from Easy to Hard, with 95% intervals from a bootstrap that resamples TASKS
def difficulty_sensitivity(d: dict[str, pd.DataFrame], reps: int = 2000, seed: int = 42) -> dict | None:
    if not _has_tiers(d):
        return None
    print_section(f"{vs(d)}: Difficulty Sensitivity, Easy to Hard")
 
    #concatenate the task id and difficulty tier columns for each strategy and drop duplicates
    tiers = (pd.concat([df[["task_id", "difficulty_tier"]] for df in d.values()]).drop_duplicates("task_id").set_index("task_id")["difficulty_tier"])
    #set the random number generator seed
    rng = np.random.default_rng(seed)
    #per tier: task ids, and each strategy's successes / episodes per task aligned to them
    ids = {t: tiers[tiers == t].index.to_numpy() for t in ("Easy", "Hard")}
    #initialize the arrays dictionary
    arrays = {}
    #loop through each strategy
    for s, df in d.items():
        #group the dataframe by task id and calculate the sum and count of successes
        per = df.groupby("task_id")["success"].agg(["sum", "count"])
        #append the sum and count of successes for each task id to the arrays dictionary
        arrays[s] = {t: (per["sum"].reindex(ids[t]).fillna(0).to_numpy(), per["count"].reindex(ids[t]).fillna(0).to_numpy()) for t in ids}
    draws = {t: rng.integers(0, len(ids[t]), size=(reps, len(ids[t]))) for t in ids}
    #initialize the point and boot dictionaries
    point, boot = {}, {}
    #loop through each strategy
    for s in d:
        #calculate the success rate for each task id
        sr = {t: arrays[s][t][0].sum() / arrays[s][t][1].sum() for t in ids}
        #append the success rate for each task id to the point dictionary
        point[s] = (sr["Easy"] - sr["Hard"], sr["Hard"] / sr["Easy"] if sr["Easy"] else np.nan)
        #initialize the b dictionary
        b = {}
        for t in ids:
            k, n = arrays[s][t]
            b[t] = k[draws[t]].sum(axis=1) / np.maximum(n[draws[t]].sum(axis=1), 1)
        boot[s] = (b["Easy"] - b["Hard"], np.where(b["Easy"] > 0, b["Hard"] / np.where(b["Easy"] > 0, b["Easy"], 1), np.nan))
    #helper function to calculate the confidence interval
    def ci(a):
        lo, hi = np.nanpercentile(a, [2.5, 97.5])
        return f"{lo:.2f} to {hi:.2f}"
    #initialize the rows list
    rows = []
    for s in d:
        #append the data for the current strategy to the rows list
        rows.append({"strategy": s,
                     "drop_Easy_to_Hard (pp)": f"{100 * point[s][0]:.1f}",
                     "95%CI (pp)": ci(100 * boot[s][0]),
                     "Hard/Easy ratio": f"{point[s][1]:.2f}",
                     "95%CI": ci(boot[s][1])})
    print(pd.DataFrame(rows).set_index("strategy").to_string())
    print("\n difference in Hard/Easy ratio between strategies (paired bootstrap):")
    #loop through each pair of strategies
    for x, y in present_pairs(d):
        #calculate the difference in the Hard/Easy ratio
        diff = boot[y][1] - boot[x][1]
        #calculate the confidence interval
        lo, hi = np.nanpercentile(diff, [2.5, 97.5])
        #calculate the verdict based on the confidence interval and print the result
        verdict = "differs" if (lo > 0 or hi < 0) else "no clear difference"
        print(f"{y} - {x}: {point[y][1] - point[x][1]:+.2f}  (95% CI {lo:+.2f} to {hi:+.2f})  -> {verdict}")
    return {"point": point, "boot": boot}

#failure-mode mix per strategy and tier, as shares of failures, with a chi-square per strategy
def failure_modes_by_tier(d: dict[str, pd.DataFrame]) -> None:
    if not _has_tiers(d):
        return
    print_section(f"{vs(d)}: Failure Modes by Difficulty Tier (RQ2)")
    try:
        #import the chi2_contingency function from scipy.stats
        from scipy.stats import chi2_contingency
    except ImportError:
        chi2_contingency = None
    #loop through each strategy
    for s, df in d.items():
        #get the failed episodes
        fails = df[df["success"] == False]
        #calculate the contingency table
        counts = (pd.crosstab(fails["difficulty_tier"], fails["termination_reason"].map(outcome_group)).reindex(index=TIER_ORDER, fill_value=0))
        #get the outcome columns
        counts = counts[[g for g in OUTCOME_ORDER[1:] if g in counts.columns]]
        shares = counts.div(counts.sum(axis=1), axis=0)
        shares.insert(0, "failures", counts.sum(axis=1))
        shares.columns.name = None
        shares.index.name = "tier"
        print(f" Strategy {s}")
        print(shares.to_string(float_format=lambda x: f"{x:.0%}" if x <= 1 else f"{x:.0f}"))
        c = counts.loc[counts.sum(axis=1) > 0, counts.sum(axis=0) > 0]
        if chi2_contingency and c.shape[0] > 1 and c.shape[1] > 1:
            chi2, p, dof, _ = chi2_contingency(c.values)
            print(f" chi-square: chi2={chi2:.1f}, dof={dof}, p={p:.4f}  -> "
                  f"{'the failure mix changes with difficulty' if p < 0.05 else 'no evidence the mix changes with difficulty'}")
        print()

#failure mode shift analysis for each strategy
def failure_mode_shift(d: dict[str, pd.DataFrame]) -> None:
    print_section(f"{vs(d)}: Failure Mode Shift")
    for s, df in d.items():
        fails = df[df["success"] == False]
        if fails.empty:
            continue
        groups = fails["termination_reason"].map(outcome_group).value_counts()
        n = len(fails)
        print(f" Strategy {s}: {n} failures")
        for g in OUTCOME_ORDER[1:]:
            if groups.get(g, 0):
                print(f"{g:18s} {groups[g]:>5}  ({groups[g] / n:.0%})")
        print()

#failure mode shift analysis for each strategy
def failure_mode_shift(d: dict[str, pd.DataFrame]) -> None:
    print_section(f"{vs(d)}: Failure Mode Shift")

    for s, df in d.items():
        fails = df[df["success"] == False]
        if fails.empty:
            continue
        groups = fails["termination_reason"].map(outcome_group).value_counts()
        n = len(fails)
        print(f" Strategy {s}: {n} failures")
        for g in OUTCOME_ORDER[1:]:
            if groups.get(g, 0):
                print(f"{g:18s} {groups[g]:>5}  ({groups[g] / n:.0%})")
        print()


# Plots for the comparison
#helper function to save the figures
def _finish(fig, ax, out_dir: Path, name: str, legend: bool = True) -> None:
    if legend:
        ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    path = out_dir / name
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")
    plt.close(fig)

#plot the cost vs success rate curve
def plot_cost_curves_overlay(d: dict, out_dir: Path, tag: str) -> None:
    fig, ax = plt.subplots()
    budgets = sorted(set().union(*[set(df["budget_level"]) for df in d.values()]))
    for s, df in d.items():
        g = df.groupby("budget_level")["success"].agg(["sum", "count"])
        sr = g["sum"] / g["count"]
        lo, hi = zip(*[wilson(int(k), int(n)) for k, n in zip(g["sum"], g["count"])])
        ax.fill_between(g.index, lo, hi, color=COLOURS[s], alpha=0.12, linewidth=0)
        ax.plot(g.index, sr, MARKERS[s], color=COLOURS[s], linewidth=2, markersize=8, label=NAMES[s])
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Success Rate (shaded: 95% interval)")
    ax.set_title(f"Success Rate vs Token Budget: {vs(d)}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels(budget_labels(budgets))
    ax.minorticks_off()
    ax.set_ylim(bottom=0)
    _finish(fig, ax, out_dir, f"cost_curve_{tag}.png")

#plot the cost vs performance curve
def plot_cost_performance(d: dict, out_dir: Path, tag: str) -> None:
    fig, ax = plt.subplots()
    notes = []
    for s, df in d.items():
        g = df.groupby("budget_level").agg(sr=("success", "mean"), tok=("total_tokens", "mean"))
        zero = g[g["tok"] <= 0]
        if len(zero):
            notes.append(f"{s} at " + ", ".join(f"{b // 1000}k" for b in zero.index) + ": no tokens spent (cannot afford a single step), 0% success")
        g = g[g["tok"] > 0]
        ax.plot(g["tok"], g["sr"], MARKERS[s], color=COLOURS[s], linewidth=2, markersize=8, label=NAMES[s])
        #label the budget beside each point, in muted text
        for budget, row in g.iterrows():
            ax.annotate(f"{budget // 1000}k", (row["tok"], row["sr"]), textcoords="offset points", xytext=(6, -12), fontsize=8, color="#52514e")
    ax.set_xscale("log")
    ticks = [t for t in [1000, 2000, 4000, 8000, 16000, 32000, 64000] if ax.get_xlim()[0] <= t <= ax.get_xlim()[1]]
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t // 1000}k" for t in ticks])
    ax.minorticks_off()
    ax.set_xlabel("Mean Tokens Actually Spent per Episode (point labels: budget)")
    ax.set_ylabel("Success Rate")
    ax.set_title(f"Cost vs Performance: {vs(d)}")
    ax.set_ylim(bottom=0)
    if notes:
        ax.text(0.99, 0.02, "\n".join(notes), transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#52514e")
    _finish(fig, ax, out_dir, f"cost_performance_{tag}.png")

def plot_per_step_cost_comparison(d: dict, out_dir: Path, tag: str) -> None:
    data, labels = [], []
    for s, df in d.items():
        v = df[df["steps"] > 0]
        data.append((v["total_tokens"] / v["steps"]).values)
        labels.append(f"Strategy {s}")

    fig, ax = plt.subplots()
    bp = ax.boxplot(data, tick_labels=labels, patch_artist=True, showfliers=False)
    for patch, s in zip(bp["boxes"], d):
        patch.set_facecolor(COLOURS[s])
        patch.set_alpha(0.5)
    ax.set_ylabel("Tokens per Step")
    ax.set_title(f"Per-Step Token Cost: {vs(d)}")
    _finish(fig, ax, out_dir, f"per_step_cost_{tag}.png", legend=False)


def plot_token_efficiency(d: dict, out_dir: Path, tag: str) -> None:
    fig, ax = plt.subplots()
    budgets = sorted(set().union(*[set(df["budget_level"]) for df in d.values()]))
    for s, df in d.items():
        g = df.groupby("budget_level").agg(sr=("success", "mean"), tok=("total_tokens", "mean"))
        eff = (g["sr"] / g["tok"] * 1000).where(g["tok"] > 0, 0)
        ax.plot(g.index, eff, MARKERS[s], color=COLOURS[s], linewidth=2, markersize=8, label=NAMES[s])
    ax.set_xlabel("Token Budget")
    ax.set_ylabel("Token Efficiency (SR per 1k tokens)")
    ax.set_title(f"Token Efficiency by Budget Level: {vs(d)}")
    ax.set_xscale("log", base=2)
    ax.set_xticks(budgets)
    ax.set_xticklabels(budget_labels(budgets))
    ax.minorticks_off()
    _finish(fig, ax, out_dir, f"token_efficiency_{tag}.png")


def plot_outcome_mix(d: dict, out_dir: Path, tag: str) -> None:
    shares = {}
    for s, df in d.items():
        counts = df["termination_reason"].map(outcome_group).value_counts()
        shares[s] = counts / counts.sum()
    groups = [g for g in OUTCOME_ORDER if any(shares[s].get(g, 0) for s in d)]

    fig, ax = plt.subplots(figsize=(10, 1.4 + 0.9 * len(d)))
    y = np.arange(len(d))
    left = np.zeros(len(d))
    for g in groups:
        vals = np.array([shares[s].get(g, 0) for s in d])
        ax.barh(y, vals, left=left, color=OUTCOME_COLOURS[g], edgecolor="white", linewidth=2, label=g)
        for yi, (v, l) in enumerate(zip(vals, left)):
            if v >= 0.06:
                ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=9, color="white")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([f"Strategy {s}" for s in d])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of Episodes")
    ax.set_title(f"How Episodes End: {vs(d)}")
    ax.legend(ncol=len(groups), loc="upper center", bbox_to_anchor=(0.5, -0.35 if len(d) > 2 else -0.45), frameon=False, fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    path = out_dir / f"outcome_mix_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved: {path}")
    plt.close(fig)


#plot the success by difficulty tier, all budgets pooled, with 95% intervals
def plot_difficulty_comparison(d: dict, out_dir: Path, tag: str) -> None:
    if not all("difficulty_tier" in df.columns for df in d.values()):
        return
    fig, ax = plt.subplots()
    width = 0.8 / len(d)
    x = np.arange(len(TIER_ORDER))
    for i, (s, df) in enumerate(d.items()):
        srs, errs = [], [[], []]
        for tier in TIER_ORDER:
            g = df[df["difficulty_tier"] == tier]["success"]
            k, n = int(g.sum()), len(g)
            sr = k / n if n else 0
            lo, hi = wilson(k, n)
            srs.append(sr)
            errs[0].append(max(0, sr - lo) if n else 0)
            errs[1].append(max(0, hi - sr) if n else 0)
        ax.bar(x + (i - (len(d) - 1) / 2) * width, srs, width * 0.92, color=COLOURS[s], yerr=errs, capsize=3, error_kw={"elinewidth": 1, "ecolor": "#52514e"}, label=NAMES[s])
    ax.set_xticks(x)
    ax.set_xticklabels(TIER_ORDER)
    ax.set_xlabel("Difficulty Tier")
    ax.set_ylabel("Success Rate (all budgets)")
    ax.set_title(f"Success by Difficulty Tier: {vs(d)}")
    _finish(fig, ax, out_dir, f"difficulty_{tag}.png")

def plot_difficulty_by_budget(d: dict, out_dir: Path, tag: str) -> None:
    if not _has_tiers(d):
        return
    budgets = sorted(set().union(*[set(df["budget_level"]) for df in d.values()]))
    top = max(df.groupby(["difficulty_tier", "budget_level"])["success"].mean().max() for df in d.values())
    for tier in TIER_ORDER:
        fig, ax = plt.subplots()
        for s, df in d.items():
            g = df[df["difficulty_tier"] == tier].groupby("budget_level")["success"].agg(["sum", "count"])
            if g.empty:
                continue
            lo, hi = zip(*[wilson(int(k), int(n)) for k, n in zip(g["sum"], g["count"])])
            ax.fill_between(g.index, lo, hi, color=COLOURS[s], alpha=0.10, linewidth=0)
            ax.plot(g.index, g["sum"] / g["count"], MARKERS[s], color=COLOURS[s], linewidth=2, markersize=8, label=NAMES[s])
        ax.set_xscale("log", base=2)
        ax.set_xticks(budgets)
        ax.set_xticklabels(budget_labels(budgets))
        ax.minorticks_off()
        ax.set_ylim(0, min(1, top * 1.35))
        ax.set_xlabel("Token Budget")
        ax.set_ylabel("Success Rate (shaded: 95% interval)")
        ax.set_title(f"{tier} Tasks: Success Rate vs Token Budget, {vs(d)}")
        _finish(fig, ax, out_dir, f"difficulty_{tier.lower()}_by_budget_{tag}.png")


#plot the outcome mix by difficulty tier
#how episodes end, per strategy and tier, as shares of 100%
def plot_outcome_mix_by_tier(d: dict, out_dir: Path, tag: str) -> None:
    if not _has_tiers(d):
        return
    labels, shares = [], []
    for s, df in d.items():
        for tier in TIER_ORDER:
            g = df[df["difficulty_tier"] == tier]
            counts = g["termination_reason"].map(outcome_group).value_counts()
            labels.append(f"{s}  {tier}")
            shares.append(counts / counts.sum() if counts.sum() else counts)
    groups = [g for g in OUTCOME_ORDER if any(sh.get(g, 0) for sh in shares)]

    #a gap between strategies so the three blocks read as groups
    y = np.array([i + (i // len(TIER_ORDER)) * 0.6 for i in range(len(labels))])
    fig, ax = plt.subplots(figsize=(10, 1.6 + 0.5 * len(labels)))
    left = np.zeros(len(labels))
    for g in groups:
        vals = np.array([sh.get(g, 0) for sh in shares])
        ax.barh(y, vals, left=left, height=0.85, color=OUTCOME_COLOURS[g], edgecolor="white", linewidth=2, label=g)
        for yi, v, l in zip(y, vals, left):
            if v >= 0.07:
                ax.text(l + v / 2, yi, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white")
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Share of Episodes")
    ax.set_title(f"How Episodes End by Difficulty Tier: {vs(d)}")
    ax.legend(ncol=len(groups), loc="upper center", bbox_to_anchor=(0.5, -0.08 - 1.2 / len(labels)), frameon=False, fontsize=9)
    ax.grid(False)
    fig.tight_layout()
    path = out_dir / f"outcome_mix_by_tier_{tag}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"saved: {path}")
    plt.close(fig)


#plot the difficulty sensitivit
#share of Easy-tier success each strategy keeps on Hard tasks, with bootstrap 95% CI
def plot_difficulty_sensitivity(d: dict, sens: dict | None, out_dir: Path, tag: str) -> None:
    if not sens:
        return
    fig, ax = plt.subplots(figsize=(10, 1.8 + 0.7 * len(d)))
    for i, s in enumerate(d):
        ratio = sens["point"][s][1]
        lo, hi = np.nanpercentile(sens["boot"][s][1], [2.5, 97.5])
        ax.errorbar(ratio, i, xerr=[[max(0, ratio - lo)], [max(0, hi - ratio)]], fmt=MARKERS[s][0], color=COLOURS[s], markersize=10, capsize=4, elinewidth=2, label=NAMES[s])
        ax.annotate(f"{ratio:.2f}", (ratio, i), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9, color="#52514e")
    ax.axvline(1.0, color="#9E9E9E", linewidth=1, linestyle=":")
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels([f"Strategy {s}" for s in d])
    #top-down order, with headroom so the value labels clear the title
    ax.set_ylim(len(d) - 0.5, -0.8)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("Hard-tier success as a share of Easy-tier success (1.0 = no drop; 95% bootstrap interval)")
    ax.set_title(f"Difficulty Sensitivity: {vs(d)}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    path = out_dir / f"difficulty_sensitivity_{tag}.png"
    fig.savefig(path, dpi=150)
    print(f"saved: {path}")
    plt.close(fig)

# Main function to run the comparison
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--c", default=None)
    ap.add_argument("--tiers", default=None)
    ap.add_argument("--out", default="../data/processed/6_budgets_ALL_tasks_decontaminted_batches/comparisons")
    args = ap.parse_args()

    paths = {"A": args.a, "B": args.b, "C": args.c}
    d = {s: load(p) for s, p in paths.items() if p}
    print("Loaded " + ", ".join(f"{len(df)} {s}" for s, df in d.items()) + " episodes")

    if args.tiers:
        d = {s: join_tiers(df, args.tiers) for s, df in d.items()}

    #restrict to tasks present in every loaded strategy, so the comparison is like for like
    shared = set.intersection(*[set(df["task_id"]) for df in d.values()])
    if len(shared) < max(df["task_id"].nunique() for df in d.values()):
        print(f"\n[compare] restricting to {len(shared)} tasks present in ALL datasets (" + ", ".join(f"{s} has {df['task_id'].nunique()}" for s, df in d.items()) + ")")
        d = {s: df[df["task_id"].isin(shared)] for s, df in d.items()}

    comparison_table(d)
    paired_tests(d)
    equivalent_budget(d)
    cost_ratio(d)
    router_vs_fixed_policies(d)
    task_level_comparison(d)
    failure_mode_shift(d)
    difficulty_table(d)
    paired_tests_by_tier(d)
    sens = difficulty_sensitivity(d)
    failure_modes_by_tier(d)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "_vs_".join(s.lower() for s in d)

    print_section("Comparison Plots")
    plot_cost_curves_overlay(d, out_dir, tag)
    plot_cost_performance(d, out_dir, tag)
    plot_per_step_cost_comparison(d, out_dir, tag)
    plot_token_efficiency(d, out_dir, tag)
    plot_outcome_mix(d, out_dir, tag)
    plot_difficulty_comparison(d, out_dir, tag)
    plot_difficulty_by_budget(d, out_dir, tag)
    plot_outcome_mix_by_tier(d, out_dir, tag)
    plot_difficulty_sensitivity(d, sens, out_dir, tag)


if __name__ == "__main__":
    main()