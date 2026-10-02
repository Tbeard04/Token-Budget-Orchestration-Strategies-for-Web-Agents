# src/router_c

This module builds Strategy C's router: the training data, the training itself, and the analysis of the trained models. The router learns offline from logged Strategy A and Strategy B episodes, and has two small neural networks ("heads"):

- **Mode head:** decides once per episode whether to run `execute` (single agent, A) or `cycle` (Planner --> Executor --> Critic, B).
- **Stop head:** decides at every step whether to `continue` or `stop` the episode early to save tokens.

| Order | File | Purpose |
|---|---|---|
| 1 | `build_transitions.py` | Turn the decontaminated A+B episodes into `mode_pairs.jsonl` (mode head data) and `transitions.jsonl` (stop head data) |
| 2 | `train/train_router.py` | Train one router version (both heads) with advantage-weighted regression and 5-fold task-level cross-validation. It saves the model, its metrics and the held-out predictions |
| 3 | `model_analysis/compare_models.py` | Plot models 1–3 against each other and against the always-A, always-B and oracle baselines |

**Run every command from `src/`** using `python -m`. `train/` and `model_analysis/`.

---

## 1. build_transitions.py

```powershell
python -m router_c.build_transitions --episodes ../data/processed/6_budgets_ALL_tasks_decontaminated_batch/final_episodes.jsonl --out-dir ../data/processed/router_c_models
```

| Argument | Default | Meaning |
|---|---|---|
| `--episodes` | `../data/processed/6_budgets_ALL_tasks_decontaminated_batch/final_episodes.jsonl` | The **combined A+B** output of `clean_contamination/join_metadata.py`. Strategy A is labelled `execute` and B is labelled `cycle`. Do not include C's episodes |
| `--raw` | A and B raw batch files | Only used if `--episodes` has no `step_log` (built without `--keep-steps`). In that case the step logs are restored from these raw files, matched on (strategy, task_id, budget). Takes one or more paths separated by spaces |
| `--out-dir` | `../data/processed/router_c_models` | Folder where `transitions.jsonl` and `mode_pairs.jsonl` are written |

**Outputs:**
- **`mode_pairs.jsonl`** (main dataset for the mode head): one row per (task, budget), with A's and B's success and tokens side by side, the task features and `both_eligible`. Only rows where both are eligible are used for training.
- **`transitions.jsonl`** (stop head): one row per website step of every episode, plus a final `stop` row. Each row records the state at that step: step index, fraction of budget left, whether the last action errored, whether the URL changed, the number of consecutive errors, and the mode. It also carries the episode's final success and tokens.

**Printed:**
- the episode count
- transitions by action (`execute`, `cycle`, `stop`)
- the number of mode pairs, and how many are eligible on both sides
- how many episodes had no steps

Expected counts for the main batch: 2,850 mode pairs (2,560 eligible on both sides) and 23,801 transitions.

---

## 2. train/train_router.py

```powershell
python -m router_c.train.train_router --model 1 --alpha <α for model 1>
python -m router_c.train.train_router --model 2 --alpha <α for model 2>
python -m router_c.train.train_router --model 3 --alpha <α for model 3>
```

Once a model has been trained, its `config.json` stores its α. Re-running `--model N` without `--alpha` reproduces it exactly.

| Argument | Default | Meaning |
|---|---|---|
| `--model` | *none (always pass it)* | Model number. Everything is written to `src/router_c/train/model_<N>/`. If left out, the folder becomes `model_None` |
| `--alpha` | `0.05`, or the value in `model_<N>/config.json` | Entropy weight α, the hyperparameter that differs between the three models. A higher α keeps the policy closer to 50/50, making it less decisive. Passing it overwrites the value saved in `config.json` |
| `--pairs` | `../data/processed/router_c_models/mode_pairs.jsonl` | Mode head data from step 1 |
| `--transitions` | `../data/processed/router_c_models/transitions.jsonl` | Stop head data from step 1 |

| Setting | Value | Meaning |
|---|---|---|
| `beta` | 0.5 | Temperature of the advantage weights `exp(A/β)` (clipped at 20) |
| `lambda` | 0.5 | Token-cost weight in both reward functions |
| `hidden` | 64 | Size of each of the two hidden layers in each MLP |
| `epochs`, `lr`, `batch` | 60, 1e-3, 256 | Adam optimiser settings |
| `folds` | 5 | Task-level cross-validation folds. All of a task's rows stay in the same fold |
| `seed` | 0 | Seed for the fold split and torch |
| `min_success_kept` | 0.90 | The suggested stop threshold is the one that saves the most tokens while keeping at least 90% of always-A's successes |
| `threshold_quantiles` | 0.30 … 0.99 | Stop thresholds swept on the held-out P(stop). This is an evaluation setting and is never saved to `config.json` |

**What it does:**
1. It builds the rewards:
   - mode: `success × (1 − λ·tokens/budget)`
   - stop: continuing earns `success − λ·remaining/budget`, while stopping earns 0
2. It trains each head on 4 folds and predicts the held-out fold, so every prediction comes from a model that never saw that task.
3. It replays the logged episodes with those held-out predictions to trace a cost/success frontier and pick the suggested threshold.
4. It retrains both heads on all the data for the live runner.

**Outputs (written to `src/router_c/train/model_<N>/`):**
- **`router.pt`:** both networks' weights, the feature normalisation, the feature names and the `stop_threshold`. This is the file `strategy_c.py` loads.
- **`metrics.json`:**
  - mode head: picks-cycle rate, AUC on the better arm, AUC where outcomes differ
  - stop head: AUC against eventual failure, mean P(stop) by tier
  - the baselines, the frontier, the suggested threshold and per-budget results
- **`oof_predictions.jsonl`:** the held-out P(cycle) and P(stop) for each (task, budget). `compare_models.py` reads this.
- **`config.json`:** the settings used for this model.

---

## 3. model_analysis/compare_models.py

```powershell
python -m router_c.model_analysis.compare_models --models ../data/processed/strategy_c_models --out ../data/processed/diagrams/strategy_c_model123_plots --chosen 2
```

| Argument | Default | Meaning |
|---|---|---|
| `--models` | `../data/processed/strategy_c_models` | Folder containing `model_1/`, `model_2/` and `model_3/`, each with `metrics.json` and `oof_predictions.jsonl`. Missing models are skipped. Training writes them to `router_c/train/`, so either copy them here or pass `--models router_c/train` |
| `--out` | `../data/processed/diagrams/strategy_c_model123_plots` | Folder for the PNGs (300 dpi) |
| `--chosen` | `2` | The model that gets single-model plots (the one deployed as Strategy C) |

**Saved:**

| File | Shows |
|---|---|
| `frontier.png` | Successes kept vs tokens saved against always-A, for every threshold of every model, with the always-A, always-B and oracle points |
| `mode_p_cycle.png` | Histogram of held-out P(cycle) per model, with each model's picks-cycle rate |
| `mode_by_winner.png` | P(cycle) grouped by which arm actually won (A only, B only, both, neither), with the AUC (`--chosen` model) |
| `stop_by_step.png` | Mean P(stop) at each step of the execute arm, per model (steps with at least 30 samples) |
| `stop_step0.png` | P(stop) at step 0 by budget and difficulty tier (`--chosen` model) |
| `per_budget_successes.png`, `per_budget_tokens.png` | Router vs always-A at each budget, with the % of tokens saved (`--chosen` model) |