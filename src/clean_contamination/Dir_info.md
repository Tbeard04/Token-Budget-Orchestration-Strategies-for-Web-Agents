# src/clean_contamination

These scripts turn the raw batch logs into clean episode files for analysis and router training. WebArena's sites keep their state between episodes. So once an agent has completed a task that writes to a site, later attempts at that task may find the change already made. This module labels each task with a risk level and drops the rows that crashed. It then flags any episode after a strategy's first success on a writing task, and attaches the difficulty labels.


| Order | File | Purpose |
|---|---|---|
| 1 | `remove_errors.py` | Remove episodes that never finished because the environment failed (crash, timeout, API error) |
| 2 | `sort_batch.py` | Put episodes back in task order after crashes and resumes, and report late arrivals |
| 3 | `classify_tasks.py` | Give every task a risk level: `read_only`, `idempotent` or `non_idempotent` |
| 4 | `flag_episodes.py` | Apply the first-success rule and set `use_for_reward` on every episode |
| 5 | `join_metadata.py` | Attach the difficulty tier and risk reason, then write `final_episodes*.jsonl` |

**Run every command from `src/`** using `python -m clean_contamination.<script>` (no `.py`).

All the commands below use these path variables (PowerShell). Adjust them if your folders differ:

```powershell
$R = "../data/raw/6_budgets_ALL_tasks_batch"
$F = "../data/processed/6_budgets_ALL_tasks_decontaminated_batch"
$M = "../data/processed/final_annotation_difficulty_tiers/task_metadata.jsonl"
$K = "../data/processed/task_list/task_risk_levels.jsonl"
```

---

## 1. remove_errors.py

```powershell
python -m clean_contamination.remove_errors --file $R/batch_Strategy_A/strategy_a.jsonl
python -m clean_contamination.remove_errors --file $R/batch_Strategy_B/strategy_b.jsonl
python -m clean_contamination.remove_errors --file $R/batch_Strategy_C/strategy_c.jsonl
```

| Argument | Default | Meaning |
|---|---|---|
| `--file` | *required* | The batch JSONL to clean. It is **edited in place** |
| `--backup` | `<file>.bak` | Where to save a copy of the original before editing |
| `--yes` | off | Skip the `Proceed? [y/N]` prompt |

The script only removes rows that have an episode-level `"error"` key, meaning the episode never completed. An `action_error` inside `step_log` is a normal failed action and is kept. Before anything is removed, it shows the errors grouped by cause (page-load timeout, Chromium crash, container not running, model/API not found, other), by strategy and by budget. It also warns if an error row contains real step or token data.

Run it **before** resuming a batch, so the removed tasks are re-run. Then run it again on the finished file. It works on any episode JSONL, including `schemes.jsonl`.

---

## 2. sort_batch.py

```powershell
python -m clean_contamination.sort_batch --file $R/batch_Strategy_A/strategy_a.jsonl
python -m clean_contamination.sort_batch --file $R/batch_Strategy_B/strategy_b.jsonl
python -m clean_contamination.sort_batch --file $R/batch_Strategy_C/strategy_c.jsonl
```

| Argument | Default | Meaning |
|---|---|---|
| `--file` | *required* | The batch JSONL to sort |
| `--out` | `None` | Write the sorted rows to a new file. If left out, the input file is sorted in place and a `.bak` copy is kept |
| `--gap-minutes` | `30` | A gap between two episodes of the same task longer than this marks the second one as a "late arrival" (a budget finished after a crash or resume) |

Episodes are grouped by task. Within a task they are ordered by timestamp, and the task blocks are ordered by when each task first appeared. The script prints the late arrivals, how many rows moved, and the first three task blocks.

---

## 3. classify_tasks.py

```powershell
python -m clean_contamination.classify_tasks --self-test
python -m clean_contamination.classify_tasks --annotations $M --output $K --configs task_analysis/test.raw.json
```

| Argument | Default | Meaning |
|---|---|---|
| `--annotations` | `../data/processed/final_annotation_difficulty_tiers/task_metadata.jsonl` | Annotated task list. Supplies each task's `intent`, `task_category` and `site` |
| `--output` | `../data/processed/task_list/task_risk_levels.jsonl` | Where the risk levels are written (one row per task) |
| `--configs` | `task_analysis/test.raw.json` | WebArena's raw task configs. These are used to find "impossible" tasks, whose expected answer is only `N/A`. Those tasks write nothing, so they are forced to `read_only`. If the file is missing, this check is skipped with a warning |
| `--show` | `5` | Number of example tasks printed per risk level. `0` hides the examples |
| `--self-test` | off | Run the built-in test cases (intent phrasing, quoted payloads, word boundaries, the N/A detector) and exit without writing anything |

**How a task is classified:** the script looks for write verbs in the task intent ("delete", "update", "post" and so on), ignoring any quoted text. It also looks for read-only phrasing ("show me", "how many") and idempotent actions (wishlist, cart, upvote). Each task gets one of three levels:
- `read_only`: can never be contaminated.
- `idempotent`: repeating the action changes nothing further, but it still follows the first-success rule unless `--lenient-idempotent` is passed in step 4.
- `non_idempotent`: always follows the first-success rule.

**Printed:** the count per risk level, a breakdown by site, how each category maps to a risk level, and the examples.

---

## 4. flag_episodes.py

```powershell
python -m clean_contamination.flag_episodes --self-test

# A and B together (router training and analysis)
python -m clean_contamination.flag_episodes --episodes $R/batch_Strategy_A/strategy_a.jsonl $R/batch_Strategy_B/strategy_b.jsonl --risk $K --metadata $M --out $F/episodes_flagged.jsonl

# C on its own
python -m clean_contamination.flag_episodes --episodes $R/batch_Strategy_C/strategy_c.jsonl --risk $K --metadata $M --out $F/batch_strategy_C/episodes_flagged_c.jsonl
```

| Argument | Default | Meaning |
|---|---|---|
| `--episodes` | A and B raw files | One or more batch JSONLs, separated by spaces. Error rows are skipped, and the number skipped is reported |
| `--risk` | `../data/processed/task_list/task_risk_levels.jsonl` | Output of step 1. Tasks missing from it count as `unknown`, which is never exempt |
| `--metadata` | `…/final_annotation_difficulty_tiers/task_metadata.jsonl` | Interaction count and difficulty tier, used only for the `suspect_pre_existing` warning. If the file is missing, every one-step first success is flagged as suspect |
| `--out` | `../data/processed/6_budgets_ALL_tasks_decontaminated_batch/episodes_flagged.jsonl` | Where the flagged episodes are written |
| `--lenient-idempotent` | off | Treat `idempotent` tasks as clean (exempt, like `read_only`). This is a sensitivity check only; the reported results use the strict default |
| `--pre-steps` | `1` | A first success that took this many steps or fewer, on a writing task that should take more, is flagged `suspect_pre_existing` (the site may already have been in the solved state) |
| `--example` | `None` | Parsed but not used by the script |
| `--self-test` | off | Run the built-in assertions and exit |

**The first-success rule:** episodes are grouped by **(strategy, task_id)** and sorted by timestamp. If the task is `idempotent`, `non_idempotent` or `unknown`, every episode after that strategy's first success gets `use_for_reward = False`. Because the grouping includes the strategy, one strategy's success never excludes another's episodes. That means running C on its own gives exactly the same flags as running it alongside A and B.

**Fields added to each episode:**
- `risk_level`, `episode_index`
- `first_success_index`, `first_success_budget`
- `suspect_pre_existing`
- `use_for_routing` (always `True`)
- `use_for_reward`
- `contamination_reason`: `post_first_success`, `idempotent_post_first_success` or `None`

**Printed:**
- reward eligibility per strategy
- exclusions by reason
- episodes by risk level
- usable vs discarded successes
- the suspect tasks

---

## 5. join_metadata.py

```powershell
# A and B: one combined file plus one file per strategy
python -m clean_contamination.join_metadata --episodes $F/episodes_flagged.jsonl --metadata $M --risk $K --out $F/final_episodes.jsonl --split --keep-steps

# C
python -m clean_contamination.join_metadata --episodes $F/batch_strategy_C/episodes_flagged_c.jsonl --metadata $M --risk $K --out $F/batch_strategy_C/final_episodes_c.jsonl --keep-steps
```

| Argument | Default | Meaning |
|---|---|---|
| `--episodes` | `../data/processed/router_c_training/episodes_flagged.jsonl` | Output of step 4. Always pass it, because this default differs from step 4's default `--out` |
| `--metadata` | `…/final_annotation_difficulty_tiers/task_metadata.jsonl` | Adds these fields by task_id: `task_category`, `pages_to_traverse`, `retrieval_type`, `interaction`, `target_locatability`, `rubric_total`, `difficulty_tier`, `label_source` |
| `--risk` | `../data/processed/task_list/task_risk_levels.jsonl` | Adds `risk_reason` (`risk_level` is already on the episode from step 4) |
| `--out` | `../data/processed/router_c_training/final_episodes.jsonl` | The combined output file |
| `--split` | off | Also write one file per strategy next to `--out`, named `<stem>_<strategy>.jsonl` (for example `final_episodes_a.jsonl` and `final_episodes_b.jsonl`) |
| `--keep-steps` | off | Keep `step_log` on each episode. Without it, `step_log` is dropped to keep the files small. **`analyse_b.py` needs it** for the Critic and per-role tables |

The `batch_analysis` commands read the split files from `batch_strategy_A/` and `batch_strategy_B/`, so move them there (or point `--file` at where they landed).

**Integrity checks printed:**
- task_ids with no metadata, no difficulty tier or no risk level
- `site` or intent mismatches between an episode and its metadata

**Tables printed:**
- the final table (episodes, tasks, successes, reward-eligible) per strategy
- exclusions by budget: the share should be roughly flat across budgets, which means the decontaminated set is safe to headline
- reward-eligible episodes per difficulty tier
- reward-eligible successes per tier
- the list of columns

---

## Data flow

```
task_metadata.jsonl --> classify_tasks --> task_risk_levels.jsonl
                                                   │
strategy_x.jsonl --> remove_errors --> sort_batch --> flag_episodes --> episodes_flagged.jsonl
                                                                            │
                                    task_metadata.jsonl + task_risk_levels --> join_metadata --> final_episodes_x.jsonl
```                              
---