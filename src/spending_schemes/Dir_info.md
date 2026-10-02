# src/spending_schemes

This module holds the RQ3 spending-scheme study. Each strategy (A, B, C) runs under three new ways of dividing a fixed **32k** budget across steps: `even`, `front_loaded` and `reactive`. The fourth scheme, `pay_as_you_go` (spend freely until the budget runs out), is how the main batch already ran. Its 32k episodes on the same tasks are reused as the reference rather than re-run.

The study uses only **read-only** tasks: each task is attempted 9 times (3 strategies × 3 schemes), and a read-only task cannot be contaminated by an earlier attempt.

| File | Type | Purpose |
|---|---|---|
| `spending_scheme_config.py` | config | Every constant behind the schemes and the task sample |
| `allowance.py` | library | How much one step may spend under each scheme, and how far to trim the prompt to fit |
| `agents.py` | library | The single agent, Planner, Executor and Critic at both `low` and `high` reasoning effort, plus each mode's instruction overhead |
| `read_only_tasks/select_tasks.py` | script | Pick the fixed sample of 100 read-only tasks, stratified by tier |
| `run_schemes.py` | script | Run the strategy × scheme × task grid against WebArena (on the AWS instance) |
| `scheme_analysis/analyse_schemes.py` | script | RQ3 tables, statistical tests and plots |

Only the three scripts are run. `run_schemes.py` imports the config, `allowance.py` and `agents.py`.

**Run every command from `src/`** using `python -m`. `read_only_tasks/` and `scheme_analysis/`.

**Order:** `select_tasks` --> `run_schemes` --> `clean_contamination.remove_errors` (then resume until there are no errors) --> `analyse_schemes`

---

## spending_scheme_config.py (edit, don't run)

| Constant | Value | Meaning |
|---|---|---|
| `BUDGET` | 32,000 | Global budget for every scheme episode |
| `STEP_BASIS` | 5 | Number of steps the budget is planned around, so the even share is 32,000 / 5 = 6,400 tokens per step |
| `SCHEMES` | 4 names | `pay_as_you_go`, `even`, `front_loaded`, `reactive` |
| `FRONT_FIRST_SHARE` | 0.40 | Share of the budget front-loaded gives to step 0 |
| `REACTIVE_STUCK_MULT` | 1.5 | Reactive multiplier after a step that errored or left the URL unchanged |
| `REACTIVE_PROGRESS_MULT` | 0.75 | Reactive multiplier after a step that changed the page without error |
| `MIN_PROMPT_CHARS` | 1,200 | A prompt is never trimmed below this, so the agent always sees the goal, URL, error line and part of the page |
| `OUTPUT_MARGIN` | low 400 / high 1,500 | Tokens reserved per call for the model's output |
| `N_TASKS`, `TIER_SPLIT` | 100: Easy 33 / Medium 33 / Hard 34 | Task sample size and how it splits across tiers |
| `TASK_SEED` | 42 | Random seed for the task sample |

## allowance.py (library)

`allowance(scheme, step, budget, spent, last_error, url_changed)` returns `(step_tokens, effort, page_share)`. At 32k this gives:

| Scheme | Step 0 | Later steps |
|---|---|---|
| `pay_as_you_go` | all remaining, low | all remaining, low |
| `even` | 6,400, low | 6,400, low (whatever remains once the 5 shares are used up) |
| `front_loaded` | 12,800, **high** | 4,800, low |
| `reactive` | 6,400, low | 9,600, **high** if the last step errored or the URL didn't change; 4,800, low if the page changed without error |

Every allowance is capped at what remains of the budget.

`trim_prompt(prompt, max_chars)` cuts from the end of the prompt, where the page's AXTree sits. That keeps the goal, URL, error and history and drops page content, then adds a `[...page truncated…]` marker. `prompt_char_limit()` is a helper version of the limit calculation; `run_schemes.py` computes the limit inline with a per-mode output reserve.


## agents.py (library)

When imported, it builds the `high`-effort copies of Strategy A's single agent and Strategy B's three agents. The `low` copies are the existing agents from `strategy_a.py` and `strategy_b.py`.
- **`AGENTS[effort][role]`:** looks up an agent by effort (`low` or `high`) and role (`single`, `planner`, `executor`, `critic`).
- **`INSTRUCTION_TOKENS`:** the instruction overhead per step: `execute` is A's instructions ÷ 4, and `cycle` is B's three instruction blocks (about 2,050 tokens).

In `cycle` mode, only the Planner switches to `high`; the Executor and Critic always run at `low`.

---

## 1. read_only_tasks/select_tasks.py

```powershell
python -m spending_schemes.read_only_tasks.select_tasks
```

There are no arguments: the paths are constants in the script, and the sample settings come from the config.

| Path | Value |
|---|---|
| `METADATA` (in) | `../data/processed/final_annotation_difficulty_tiers/task_metadata.jsonl` |
| `RISK` (in) | `../data/processed/task_list/task_risk_levels.jsonl` (from `clean_contamination/classify_tasks.py`) |
| `OUT` (out) | `../data/processed/task_list/read_only_tasks.jsonl` |

The script keeps tasks whose `risk_level` is `read_only` and drops "impossible" tasks (those whose expected answer is N/A). It then draws a seeded random sample of 33 Easy, 33 Medium and 34 Hard tasks. It stops with an error if any tier has too few eligible tasks. Each output row holds `task_id`, `site` and `difficulty_tier`. With the same seed it always returns the same 100 tasks (default seed is 42)

---

## 2. run_schemes.py (AWS instance)

The WebArena containers must be up. The tasks are read-only, so the sites never need resetting between episodes.

```bash
tmux new -s schemes
cd ~/project/src
python -u -m spending_schemes.run_schemes
```

| Argument | Default | Meaning |
|---|---|---|
| `--out` | `../data/processed/schemes_output/schemes.jsonl` | Output JSONL. **Appended to**: on restart, any (strategy, scheme, task) already in the file is skipped, so re-running the same command resumes the run |
| `--schemes` | `even front_loaded reactive` | Schemes to run, separated by spaces. `pay_as_you_go` is not run here because it comes from the main batch |
| `--strategies` | `A B C` | Strategies to run, separated by spaces. `C` loads the router through `strategy_c.configure()` |
| `--n` | `None` | Parsed but **not used**: the full task list always runs |
| `--smoke` | off | Parsed but **not used** |
| `--dry-run` | off | Parsed but **not used** |

**Not arguments:**
- The task list is `TASKS_FILE = ../data/processed/task_list/read_only_tasks.jsonl`, the output of step 1.
- The budget is `C.BUDGET`.

**Full grid:** 100 tasks × 3 schemes × 3 strategies = **900 episodes**.

**Each step of an episode:**
1. **C only:** the router may stop the episode.
2. `allowance()` sets the step's tokens and effort.
3. The prompt is trimmed to fit, after reserving output tokens: one `OUTPUT_MARGIN[effort]` for `execute`, and `OUTPUT_MARGIN[effort] + 2 × OUTPUT_MARGIN["low"]` for `cycle`.
4. The step is skipped with `budget_would_exceed` if its estimated cost would go over 32k.

**Each row records:**
- the usual episode fields, plus `scheme`, `trimmed_steps` and `tokens_by_role`
- in each `step_log` entry: `allowance`, `effort`, `prompt_chars_full`, `prompt_chars` and `trimmed`
- **C only:** the `router` block

**Errors:** if an episode fails (API 429 error, Chromium timeout and so on), the script still writes a row for it, with an `"error"` key. That row counts as done, so the episode **will not be re-run until the row is removed**:

```bash
python -m clean_contamination.remove_errors --file ../data/processed/schemes_output/schemes.jsonl
```

If many errors appear at once, check your OpenAI credits and look for leftover Chromium processes (`pkill -f chrom`) before resuming.

---

## 3. scheme_analysis/analyse_schemes.py

Run locally (PowerShell):

```powershell
$F = "../data/processed/6_budgets_ALL_tasks_decontaminated_batch"
python -m spending_schemes.scheme_analysis.analyse_schemes --schemes ../data/processed/schemes_output/schemes.jsonl --main-a $F/batch_strategy_A/final_episodes_a.jsonl --main-b $F/batch_strategy_B/final_episodes_b.jsonl --main-c $F/batch_strategy_C/final_episodes_c.jsonl --tasks ../data/processed/task_list/read_only_tasks.jsonl --reference pay_as_you_go --out ../data/processed/diagrams/schemes_output
```

| Argument | Default | Meaning |
|---|---|---|
| `--schemes` | `../data/processed/schemes_output/schemes.jsonl` | Output of `run_schemes.py`. Error rows are dropped when it loads |
| `--main-a` | `None` | Strategy A main-batch file. Only its **32k** rows on the scheme tasks are kept and labelled `pay_as_you_go` |
| `--main-b` | `None` | The same, for Strategy B |
| `--main-c` | `None` | The same, for Strategy C. Leave out all three `--main-*` arguments to compare only the three new schemes |
| `--tasks` | `../data/processed/task_list/read_only_tasks.jsonl` | Source of each task's difficulty tier. If the file is missing, the tiers are taken from the main-batch rows |
| `--reference` | `pay_as_you_go` if main files are given, otherwise `even` | The scheme every other scheme is measured against in the bootstrap effect table and plots |
| `--out` | `../data/processed/diagrams/schemes_output` | Folder for the PNGs and `grid.csv` |

**Printed:**
- Grid coverage: episodes per strategy × scheme. Expect 100 in each cell.
- Success rate (Wilson 95% intervals), mean and median tokens, tokens per success, success per 1k tokens, steps, trimmed share and high-effort share.
- Scheme vs scheme within each strategy, paired on task: exact McNemar for success and Wilcoxon signed-rank for tokens. Compare the p-values with a Bonferroni threshold of 0.05/18 = 0.0028 (6 pairs × 3 strategies).
- Effect of each scheme against the reference: change in success (percentage points) and in tokens (%), each with a 95% interval from a task-level bootstrap (2,000 resamples, seed 42).
- Strategy vs strategy under each scheme, using McNemar. Bonferroni threshold: 0.05/12 = 0.0042 (3 pairs × 4 schemes).
- Which scheme suits which strategy.
- How episodes end (outcome mix).
- Success rate by tier (descriptive only, about 33 tasks per tier).

**Saved:**

| File | Shows |
|---|---|
| `success_by_scheme.png` | Success rate with Wilson intervals, schemes × strategies |
| `tokens_by_scheme.png` | Mean tokens per episode |
| `tokens_per_success.png` | Total tokens ÷ successes (lower is better) |
| `cost_vs_success.png` | Every strategy × scheme cell: colour shows the strategy and marker shape shows the scheme |
| `cost_vs_success_a.png`, `_b.png`, `_c.png` | One zoomed, labelled version per strategy |
| `effect_success.png`, `effect_tokens.png` | Bootstrap effects against `--reference` with 95% intervals |
| `outcome_mix.png` | How episodes end, per strategy × scheme |
| `grid.csv` | The success/efficiency table as a CSV |