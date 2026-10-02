# src/batch_analysis

This module analyses the main batch, which covers 475 tasks × 6 budgets (2k–64k). It has one script per strategy, which prints tables and saves that strategy's diagrams. 
A comparison script then tests the strategies against each other for RQ1 (cost against performance) and RQ2 (difficulty). All of them share one helper library, `shared.py`.

| File | Purpose |
|---|---|
| `analyse_a.py` | Strategy A (single agent): shared analysis plus budget-binding, cost-floor, wrong-answer and budget-utilisation tables |
| `analyse_b.py` | Strategy B (Planner --> Executor --> Critic): shared analysis plus Critic, per-role cost and Planner tables |
| `analyse_c.py` | Strategy C (offline-RL router): shared analysis plus router-stop tables and plots, broken down by budget and by tier |
| `compare_abc.py` | Compares A, B and C (or A vs B alone): McNemar tests, cost ratios, router vs fixed policies, difficulty sensitivity, failure modes by tier |
| `shared.py` | The library behind every script: loading and decontamination, Wilson intervals, McNemar, shared tables and plots, colours. It is not run directly |
| `__init__.py` | Makes the folder importable, so the scripts run with `python -m` |

**Run every command from `src/`.** The scripts import `batch_analysis.shared`, so use `python -m batch_analysis.<script>` (no `.py` at the end of the script).

All the commands below use these two path variables (PowerShell):

```powershell
$F = "../data/processed/6_budgets_ALL_tasks_decontaminated_batch"
$O = "../data/processed/diagrams/6_budgets_ALL_tasks_decontaminted_batches"
```

---

## Decontamination (applies to every script)

`shared.load()` always drops rows that have an `error` value. If the file has a `use_for_reward` column (the `final_episodes_*.jsonl` files do), it also keeps only the rows where it is `True`. That removes the repeat attempts after a task's first success, so later runs of a task cannot benefit from an earlier success. The load line prints how many rows were kept and how many were excluded.

Raw batch files (`batch_strategy_*.jsonl`) have no `use_for_reward` column, so they load unfiltered. The dissertation results use the `final_episodes_*.jsonl` files.

---

## analyse_a.py

```powershell
python -m batch_analysis.analyse_a --file $F/batch_strategy_A/final_episodes_a.jsonl --out $O/strategy_a_batch
```

| Argument | Default | Meaning |
|---|---|---|
| `--file` | *required* | Strategy A episode JSONL, i.e. `final_episodes_a.jsonl` |
| `--tiers` | `None` | Path to `task_metadata.jsonl`, which joins `difficulty_tier`, `task_category` and `rubric_total` by task_id. Only needed for raw batch files, because the final_episodes files already carry the tiers (the script prints "already present" and skips the join) |
| `--out` | `../data/processed/strategy_a` | Folder for the PNGs (created if missing). Always pass the folder you actually use, because the default is not it |
| `--verbose-tasks` | off | Print the full per-task solvability table instead of the summary |

**Printed:** the shared tables (see `shared.py`), then:
- Budget as Binding Constraint
- Cost Floor Profile
- Answer Failures (wrong answers by budget and by site)
- Budget Utilisation

**Saved:** `cost_curve_a.png`, `termination_reasons_a.png`, `token_boxplot_a.png`, `success_by_site_a.png`, `difficulty_curve_a.png`

---

## analyse_b.py

```powershell
python -m batch_analysis.analyse_b --file $F/batch_strategy_B/final_episodes_b.jsonl --out $O/strategy_b_batch
```

| Argument | Default | Meaning |
|---|---|---|
| `--file` | *required* | Strategy B episode JSONL. It must still contain `step_log` (built with `join_metadata --keep-steps`), because the Critic and per-role tables read it |
| `--tiers` | `None` | As in analyse_a: only needed for raw batch files |
| `--out` | `../data/processed/strategy_b` | Folder for the PNGs |
| `--verbose-tasks` | off | Print the full per-task solvability table |
| `--list-revisions` | off | Print every Critic revision. Without this flag, only the first 20 are printed |

**Printed:** the shared tables, then:
- Critic analysis (revision rate)
- Revision state profile
- Rubber-stamp cost (tokens spent on Critic approvals that changed nothing)
- Per-role cost (Planner, Executor, Critic)
- Planner value
- Critic revision details

**Saved:** `critic_analysis_b.png`, `role_cost_breakdown_b.png`, plus the five shared plots with the suffix `_b`

---

## analyse_c.py

```powershell
python -m batch_analysis.analyse_c --file $F/batch_strategy_C/final_episodes_c.jsonl --out $O/strategy_c_batch
```

| Argument | Default | Meaning |
|---|---|---|
| `--file` | *required* | Strategy C episode JSONL, which holds the `router` block with the mode, stop step, `p_cycle` and decisions |
| `--tiers` | `None` | As in analyse_a: only needed for raw batch files |
| `--all-rows` | off | Keep contaminated rows (`use_for_reward == False`). Use it only as a sanity check, never for reported results. This is the only script with this flag |
| `--verbose-tasks` | off | Print the full per-task solvability table |
| `--out` | `…/6_budgets_ALL_tasks_decontaminted_batches/strategy_c_batch` | Folder for the PNGs |

**Printed:** the shared tables, then:
- Router by Budget: stops, the success rate over all episodes vs over attempted episodes only, and the cycle share
- Router by Difficulty Tier

**Saved:**
- `success_all_vs_attempted_c.png`
- `router_stop_rate_c.png`
- `router_stop_by_tier_c.png`
- `router_stop_step_c.png`
- `tier_all_vs_attempted_c.png`
- the five shared plots with the suffix `_c`

Note: `analyse_c` only makes sense on a Strategy C file.

---

## compare_abc.py

```powershell
# A vs B vs C (dissertation results)
python -m batch_analysis.compare_abc --a $F/batch_strategy_A/final_episodes_a.jsonl --b $F/batch_strategy_B/final_episodes_b.jsonl --c $F/batch_strategy_C/final_episodes_c.jsonl --out $O/comparisons

# A vs B only (leave out --c)
python -m batch_analysis.compare_abc --a $F/batch_strategy_A/final_episodes_a.jsonl --b $F/batch_strategy_B/final_episodes_b.jsonl --out $O/comparisons
```

| Argument | Default | Meaning |
|---|---|---|
| `--a` | *required* | Strategy A episode JSONL |
| `--b` | *required* | Strategy B episode JSONL |
| `--c` | `None` | Strategy C episode JSONL. Leave it out to compare A vs B only. This changes the plot filenames' tag from `a_vs_b_vs_c` to `a_vs_b` and skips the router vs fixed policies table |
| `--tiers` | `None` | `task_metadata.jsonl`, joined to every dataset. Only needed for raw batch files |
| `--out` | `../data/processed/6_budgets_ALL_tasks_decontaminted_batches/comparisons` | Folder for the PNGs. Pass `$O/comparisons`, because the default is missing `diagrams/` |

**Printed (RQ1, cost against performance):**
- Comparison table (success rate with Wilson 95% intervals, and tokens per budget)
- Paired tests: exact McNemar per budget and pooled, with a Bonferroni threshold of 0.017 for the 3 pairs
- Equivalent budget
- Cost ratio
- Router vs fixed policies (needs `--c`)
- Task-level comparison
- Failure mode shift

**Printed (RQ2, difficulty):**
- Difficulty table
- Paired tests by tier (Bonferroni threshold 0.0056 for 9 tests)
- Difficulty sensitivity (Hard/Easy ratio with a task-level bootstrap CI, 2,000 resamples, seed 42)
- Failure modes by tier (with chi-square)

**Saved** (`{tag}` is `a_vs_b_vs_c` or `a_vs_b`):
- `cost_curve_{tag}.png`
- `cost_performance_{tag}.png`
- `per_step_cost_{tag}.png`
- `token_efficiency_{tag}.png`
- `outcome_mix_{tag}.png`
- `difficulty_{tag}.png`
- `difficulty_easy_by_budget_{tag}.png`, `difficulty_medium_by_budget_{tag}.png`, `difficulty_hard_by_budget_{tag}.png`
- `outcome_mix_by_tier_{tag}.png`
- `difficulty_sensitivity_{tag}.png`

---