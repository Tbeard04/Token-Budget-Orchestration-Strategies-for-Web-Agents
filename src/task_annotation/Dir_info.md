# src/task_annotation

This module gives every task a difficulty tier, which RQ2 uses. Each task is scored on a four-dimension rubric (each dimension 0–2, so the total is 0–8) and banded into **Easy (0–2)**, **Medium (3–5)** or **Hard (6–8)**. An LLM annotator scores all the tasks. It is calibrated on hand-labelled exemplars and checked against held-out hand labels with Cohen's kappa. The hand labels then replace the model's scores for those tasks.

| Dimension | 0 | 1 | 2 |
|---|---|---|---|
| `pages_to_traverse` | one page | two or three pages | four or more / unbounded |
| `retrieval_type` | read one value | compare or filter a few | aggregate or count over a set |
| `interaction` | read-only | one form or click sequence | multi-step state change |
| `target_locatability` | named in the task | derivable from the page | must be discovered |

### Files

| File | Type | Purpose |
|---|---|---|
| `annotate_tasks.py` | script | Entry point with four modes: write a blank gold set, run the annotator, apply the gold labels, re-band |
| `label_tool.py` | script | Convert the gold set to and from a CSV so you can hand-label it in Excel, and check the labels for consistency |
| `cohen_kappa.py` | script | Measure agreement between the hand labels and the annotator |
| `tier_diagrams.py` | script | Plot the final tiers |
| `rubric.py` | library | The four dimensions, the annotator instructions, the `TaskAnnotation` schema, the tier cut-points (`EASY_MAX = 2`, `MEDIUM_MAX = 5`), `tier_of()`, `total_of()` and `normalise_template()` |
| `annotator.py` | library | Builds the LLM annotator (through `wa_env.make_agent`), scores one task, and loads the task categories |
| `gold_set.py` | library | Writes the blank gold set, loads hand labels, splits them into exemplars and held-out labels, and applies the gold labels |
| `observed_cost.py` | library | Summarises each task's cost from logged episodes (median and max steps, distinct URLs, median tokens), shown to the annotator as a hint |
| `postprocess.py` | library | Template harmonisation, the review queue, and re-banding |
| `reporting.py` | library | The tier, site, category, dimension, validity and review-queue summaries printed after an annotation run |
| `agreement.py` | library | Cohen's kappa (unweighted, linear, quadratic), confusion matrices, Landis & Koch bands, Pearson and Spearman. Written directly, with no dependencies |
| `tasks.py` | library | The single-site task pools, task sampling and the site-proportional allocation. It adds `src/` to the import path so `wa_env` can be found |

**Run every command from `src/task_annotation/`**, using `python <script>.py`. The scripts import their sibling files by bare name (`import tasks`, `from rubric import …`), so `python -m` from `src/` won't work. All the default paths start with `../../`.

```powershell
cd src/task_annotation
$EP = "../../data/raw/6_budgets_ALL_tasks_batch/batch_Strategy_A/strategy_a.jsonl"
$G  = "../../data/raw/annotation_tests/run2/gold_set_50.jsonl"
$P  = "../../data/raw/annotation_tests/run2/task_metadata_model.jsonl" #Model only annotation
$M  = "../../data/processed/final_annotation_difficulty_tiers/task_metadata.jsonl"
```

Set `$EP` to the episode file(s) you used; `--episodes` accepts more than one. The annotation run calls the OpenAI API, so `src/.env` must hold the key.

**Pipeline:** blank gold set → hand-label it in a CSV --> annotation run --> kappa --> apply the gold labels --> (re-band) --> diagrams

---

## 1. Write the blank gold set: `annotate_tasks.py --gold-set`

```powershell
python annotate_tasks.py --gold-set 50 --gold-out $G --episodes $EP --seed 42
```

| Argument | Default | Meaning |
|---|---|---|
| `--gold-set` | `None` | Number of tasks to sample for hand-labelling. The sample is split across sites in proportion to each site's task pool |
| `--gold-out` | `../../data/raw/annotation_tests/run2/gold_set_50.jsonl` | Where the blank rows are written. It must differ from `--out` |
| `--episodes` | `None` | Episode file(s). Each row gets an `observed_hint` (median and max steps, distinct URLs) to help you score `pages_to_traverse` |
| `--sites` | all sites in `wa_env.SITES` | Sites to sample from |
| `--seed` | `42` | Random seed for the sample |

Each row contains the task id, site, intent, the evaluation criteria (truncated to 1,200 characters), and the four dimensions set to `null`, ready to fill in. The rubric summary is printed at the end.

---

## 2. Hand-label the gold set: `label_tool.py`

```powershell
python label_tool.py --gold $G --to-csv --csv ../../data/raw/annotation_tests/run2/gold_set_50.csv
# fill in the four score columns in Excel (0, 1 or 2), save as CSV
python label_tool.py --gold $G --from-csv --csv ../../data/raw/annotation_tests/run2/gold_set_50.csv
python label_tool.py --gold $G --check
```

| Argument | Default | Meaning |
|---|---|---|
| `--gold` | *required* | The gold set JSONL from step 1 |
| `--csv` | `None` | The CSV to write (`--to-csv`) or read (`--from-csv`) |
| `--out` | `None` | `--from-csv` only: write the labelled JSONL to a new file. Without it, `--gold` is updated in place and the original is kept as `.bak` |
| `--to-csv` | - | Mode: export to CSV. Columns: task_id, site, `group`, intent, eval type and target (made readable), the observed hints, and the four blank score columns. Near-identical intents (template similarity ≥ 0.85) share a `group` id (G1, G2, …) and sit next to each other, so you can score them the same way |
| `--from-csv` | - | Mode: copy the scores from the CSV back into the gold JSONL. Rows that are blank, outside 0–2, or have unknown task ids are reported. It runs `--check` afterwards |
| `--check` | - | Mode: report how many tasks are labelled, the score distribution for each dimension, the tier split, and any **clashes** where tasks in the same group were scored differently |

Pick exactly one of `--to-csv`, `--from-csv` or `--check`.

---

## 3. Run the annotator: `annotate_tasks.py`

```powershell
python annotate_tasks.py --intents ../../data/processed/task_list/task_intents.json --episodes $EP --gold-labels $G --n-exemplars 20 --out $P
```

| Argument | Default | Meaning |
|---|---|---|
| `--tasks` | `None` | Annotate only these task ids, separated by spaces. If left out, tasks are sampled with `--n`, `--sites` and `--seed` |
| `--n` | `999` | Maximum tasks per site. 999 means every single-site task |
| `--sites` | all sites in `wa_env.SITES` | Sites to annotate |
| `--seed` | `42` | Random seed for the task sample **and** for which gold labels become exemplars |
| `--intents` | `None` | `task_intents.json` from `task_analysis/extract_task_intent.py`. Gives each row its `task_category`; without it the category is `unknown` |
| `--episodes` | `None` | Episode file(s) for the observed-cost hint added to the prompt. If several files are given, the first file to cover a task wins |
| `--gold-labels` | `None` | The hand-labelled gold set from step 2 |
| `--n-exemplars` | `20` | How many gold labels (balanced across sites) are put into the annotator's prompt as calibration examples. The rest are held out for kappa. Setting this ≥ the number of gold labels leaves nothing held out, which inflates kappa (a warning is printed) |
| `--review-threshold` | `0.75` | Rows whose confidence is below this are flagged `needs_review` |
| `--no-harmonise` | off | Skip template harmonisation. Normally, tasks with the same intent template (numbers, names and quoted text replaced by placeholders) are set to the group's median scores, and exemplar tasks are never changed |
| `--out` | `../../data/tasks/task_metadata.jsonl` | Where the annotated rows are written |

**Each output row holds:**
- the task id, site, intent and template
- the four scores, `rubric_total`, `difficulty_tier`, `task_category` and `confidence`
- `observed` (when episodes are given)
- `gold_exemplar` and `gold_heldout`
- `template_adjusted`, `needs_review` and `review_reasons`

A row is flagged for review when its confidence is low, when harmonisation changed it, or when its page score contradicts the observed URLs: scored 0 pages but 3 or more were visited, or scored 2 pages but at most 1 was visited.

**Printed:**
- one line per task (`*` = has an observed cost, `E` = exemplar, `H` = held out)
- the tier distribution, tiers by site, the category split and the score spread per dimension
- tier vs observed cost (a validity check: steps and tokens should rise with tier)
- the review queue

---

## 4. Agreement: `cohen_kappa.py`

```powershell
python cohen_kappa.py --gold $G --pred $P --out ../../data/raw/annotation_tests/run2/kappa_report.json --confusion
```

| Argument | Default | Meaning |
|---|---|---|
| `--gold` | *required* | Hand-labelled gold set. Rows that are still blank are skipped and reported |
| `--pred` | *required* | Annotator output from step 3, **before** the gold labels are applied. After step 5 the gold tasks carry the human scores, so kappa would be 1 |
| `--out` | `None` | Also save the full report (every subset, dimension and confusion matrix) as JSON |
| `--show` | `15` | Number of largest disagreements to print. `0` hides them |
| `--confusion` | off | Also print each dimension's 3×3 confusion matrix (rows = human, columns = model) |

**Printed, separately for all matched tasks, the HELD OUT tasks and the EXEMPLARS:**
- per dimension: kappa (unweighted and quadratic), exact agreement, agreement within ±1, bias (model − human), and the Landis & Koch band
- tier kappa and the tier confusion matrix
- for the rubric total: means, MAE, Pearson r and Spearman ρ

**Report the HELD OUT subset**, because the exemplars were in the annotator's prompt.

---

## 5. Apply the gold labels: `annotate_tasks.py --apply-gold`

```powershell
python annotate_tasks.py --apply-gold $P --gold-labels $G --out $M
```

| Argument | Default | Meaning |
|---|---|---|
| `--apply-gold` | `None` | The annotator output to merge into (`$P`) |
| `--gold-labels` | *required for this mode* | The hand-labelled gold set |
| `--out` | `../../data/tasks/task_metadata.jsonl` | Where the final metadata is written. Use the `final_annotation_difficulty_tiers` path, because every later module reads it |

For each gold task, the human scores replace the model's, `rubric_total` and `difficulty_tier` are recalculated, `label_source` is set to `human` and the review flags are cleared. Every other row gets `label_source = model`. The script prints how many human labels differed from the model, and the final tier split.

---

## 6. (Optional) Re-band: `annotate_tasks.py --reband`

```powershell
python annotate_tasks.py --reband $M
```

| Argument | Default | Meaning |
|---|---|---|
| `--reband` | `None` | A metadata file, **edited in place**. `rubric_total` and `difficulty_tier` are recalculated from the stored scores, without calling the LLM |

Use this after changing `EASY_MAX` or `MEDIUM_MAX` in `rubric.py`. It prints how many tiers changed and the new split.

---

## 7. Diagrams: `tier_diagrams.py`

```powershell
python tier_diagrams.py --metadata $M --out ../../data/processed/diagrams/difficulty_tiers
```

| Argument | Default | Meaning |
|---|---|---|
| `--metadata` | `../../data/processed/final_annotation_difficulty_tiers/task_metadata.jsonl` | The final metadata |
| `--out` | `../../data/processed/diagrams/difficulty_tiers` | Folder for the PNGs |

**Saved:**

| File | Shows |
|---|---|
| `tier_distribution.png` | Tasks per tier, with counts and percentages |
| `tier_by_site.png` | 100%-stacked tier composition for each site |
| `tier_by_category.png` | 100%-stacked tier composition for each task category |

---

### Where the output goes next

`task_metadata.jsonl` feeds:
- `clean_contamination/classify_tasks.py` (`--annotations`)
- `flag_episodes.py` and `join_metadata.py` (`--metadata`)
- `spending_schemes/read_only_tasks/select_tasks.py`
