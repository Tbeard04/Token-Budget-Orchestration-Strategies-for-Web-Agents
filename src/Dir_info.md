# src/ — main batch scripts

The core of the experiment: the shared environment layer, the three orchestration strategies, and the batch runner that collected every main-batch episode.

| File | Purpose |
|---|---|
| `wa_env.py` | Shared WebArena/BrowserGym layer used by every strategy: environment setup, prompt building, agent construction, guard thresholds and shared constants |
| `strategy_a.py` | Strategy A: single agent, one LLM call per step |
| `strategy_b.py` | Strategy B: fixed Planner --> Executor --> Critic pipeline, three LLM calls per step |
| `strategy_c.py` | Strategy C: router-directed; a trained router chooses A's or B's mode per episode and can stop the episode at any step |
| `run_batch.py` | Runs one strategy over many tasks and budgets, with checkpointing and resume |


## `run_batch.py` — full data collection

Runs one strategy across a set of tasks × every budget level. Each episode is written to a JSONL file the moment it finishes, so a crash never loses completed work. Re-running the **same command** resumes: episodes already in the output file (matched on strategy, task and budget) are skipped.

```bash
python -m run_batch --strategy A --n 999 --seed 42 --out ../data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_A/strategy_a.jsonl
python -m run_batch --strategy B --n 999 --seed 42 --out ../data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_B/strategy_b.jsonl
python -m run_batch --strategy C --n 1000 --seed 42 --out ../data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_C/strategy_c.jsonl
```

| Argument | Default | Meaning |
|---|---|---|
| `--strategy` | *(required)* | Which strategy to run: `A`, `B` or `C` |
| `--n` | `67` | Tasks sampled **per site** from the single-site task pool. Any value at or above the largest pool runs every single-site task |
| `--sites` | `shopping shopping_admin reddit` | Which WebArena sites to draw tasks from (space-separated) |
| `--budgets` | `2000 4000 8000 16000 32000 64000` | Token budget levels; every task is run once at each (space-separated) |
| `--seed` | `42` | Random seed for task sampling and run order. Use the same seed to reproduce the same task set and order (Default is 42) |
| `--out` | `../data/raw/strategy_<x>.jsonl` | Output JSONL file. Always **appended to**, never overwritten |
| `--router-dir` | `../data/processed/router_c_models/model_2` | **C only.** Folder containing the trained router (`router.pt`) |
| `--stop-answer` | `none` | **C only.** What happens when the router stops an episode: `none` ends it with no answer; `na` submits "N/A" first |

Strategy C only: a second file next to the output, <name>_routing.jsonl, with one line per episode recording the router's decisions (mode chosen, stop probability at every step, where it stopped).

---

## `strategy_a.py` — single agent (quick test runs)

Runs Strategy A on a few chosen tasks at one budget. Used for testing; the main batch uses `run_batch.py`.

```bash
python -m strategy_a 21 47 --budget 16000 --out ../data/raw/strategy_a_results.jsonl
```

| Argument | Default | Meaning |
|---|---|---|
| `tasks` | one task per site | WebArena task IDs to run (space-separated, no flag) |
| `--budget` | `16000` | Token budget for every episode |
| `--out` | `../data/raw/strategy_a_results.jsonl` | Output JSONL file. **Overwritten** on each run |


---

## `strategy_b.py` — Planner --> Executor --> Critic (quick test runs)

Runs Strategy B on chosen tasks at one budget. Each step makes three calls: the Planner sets a sub-goal, the Executor turns it into an action, and the Critic approves or revises it.

```bash
python -m strategy_b 21 47 --budget 16000 --out ../data/raw/strategy_b_results.jsonl
```

| Argument | Default | Meaning |
|---|---|---|
| `tasks` | *(required)* | WebArena task IDs to run (space-separated, no flag) |
| `--budget` | `16000` | Token budget for every episode |
| `--out` | `../data/raw/strategy_b_results.jsonl` | Output JSONL file. **Overwritten** on each run |

---

## `strategy_c.py` — router directed (quick test runs)

Runs Strategy C on chosen tasks at one budget. At the start of each episode the router picks A's way (`execute`) or B's way (`cycle`); before every step it decides whether to stop.

```bash
python -m strategy_c 21 47 --budget 16000 --router-dir ../data/processed/router_c_models/model_2 --stop-answer none --out ../data/raw/strategy_c_results.jsonl
```

| Argument | Default | Meaning |
|---|---|---|
| `tasks` | none | WebArena task IDs to run (space-separated, no flag). **Give at least one**: with none, nothing runs |
| `--budget` | `16000` | Token budget for every episode |
| `--router-dir` | `../data/processed/router_c_models/model_2` | Folder containing the trained router (`router.pt`) |
| `--stop-answer` | `none` | `none`: a router stop ends the episode with no answer. `na`: submits "N/A" first |
| `--out` | `../data/raw/strategy_c_results.jsonl` | Output JSONL file. **Overwritten** on each run |

**Also requires:** the router's task features are read from `../data/processed/final_annotation_difficulty_tiers/task_metadata.jsonl` and `../data/processed/task_list/task_risk_levels.jsonl`.

Each record includes a `router` block: model folder, stop threshold, mode chosen and its probability, and the stop probability at every step.