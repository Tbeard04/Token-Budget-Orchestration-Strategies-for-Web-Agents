# src/task_analysis

This module profiles the WebArena task set before any agent runs. It keeps the single-site tasks on the three sites used in this project (shopping, shopping_admin, reddit) and puts each task's intent into a category: information retrieval, navigation, create, modify value, delete, bulk action, purchase or other. It then plots how the tasks are spread across categories, sites and contamination risk.

| Order | File | Purpose |
|---|---|---|
| 1 | `extract_task_intent.py` | Read WebArena's raw task configs, keep the single-site tasks on the target sites, assign each intent a category and write `task_intents.json` |
| 2 | `task_category_plots.py` | Plot the category breakdown overall, by site and by contamination risk |

**Run every command from `src/`** using `python -m task_analysis.<script>` (no `.py`).

---

## 1. extract_task_intent.py

```powershell
python -m task_analysis.extract_task_intent --input ../data/processed/task_list/test.raw.json --output ../data/processed/task_list/task_intents.json
```

| Argument | Default | Meaning |
|---|---|---|
| `--input` | `../data/processed/task_list/test.raw.json` | WebArena's raw task configs (all 812 tasks). Both a single JSON array and one JSON object per line are accepted |
| `--output` | ``../data/processed/task_list/task_intents.json` | Where the categorised task file is written. |

**Which tasks are kept:** tasks whose `sites` list holds exactly one site, which must be `shopping`, `shopping_admin` or `reddit`. Multi-site tasks and the other sites (GitLab, map, Wikipedia) are dropped.

**How a task is categorised:** its intent, lowercased, is checked against regex groups in a fixed order. The first group that matches decides the category:

1. **Read-only phrasing** ("show me", "how many", "what is"…) --> `information_retrieval`. This is checked first, so a question stays a question even if it mentions "post" or "purchase".
2. **Browsing** ("browse", "search for", "find"…) --> `navigation`
3. "like all", "upvote", "thumbs up"… --> `bulk_action`
4. "delete", "remove" --> `delete`. Word boundaries stop "remover" from matching.
5. "change", "update", "reduce", "disable"… --> `modify_value`
6. "buy", "checkout", "purchase" --> `purchase`
7. "post", "create", "add", "reply", "subscribe"… --> `create`
8. Anything else --> `other`

**Output (`task_intents.json`):**

| Key | Contents |
|---|---|
| `summary` | `total_tasks`, plus task counts `by_site` and `by_category` |
| `task_categories` | `{"<task_id>": "<category>"}` for **every** kept task. The keys are strings, so convert them back with `int()` when reading |
| `sites` | Per site, then per category: the task `count`, the number of `distinct_templates`, and one representative task per template (`task_id`, `intent`, `eval_type`) |

---

## 2. task_category_plots.py

```powershell
python -m task_analysis.task_category_plots --input ../data/processed/task_list/task_intents.json --out ../data/processed/diagrams/task_list
```

| Argument | Default | Meaning |
|---|---|---|
| `--input` | *required* | The `task_intents.json` written by step 1. The script reads its `summary.by_category` and `sites` sections |
| `--out` | `../../data/processed/diagrams/task_list` | Folder for the PNGs (created if missing). The default only works when run from inside `src/task_analysis/`, so pass `../data/processed/diagrams/task_list` when running from `src/` |

**Saved:**

| File | Shows |
|---|---|
| `task_categories_overall.png` | Horizontal bars of tasks per category, with counts and percentages |
| `task_categories_by_site.png` | Grouped bars of categories within each site |
| `task_categories_risk.png` | A stacked bar splitting the categories into **read-only** (information retrieval, navigation, other) and **state-change** (create, modify value, bulk action, delete, purchase), with each group's total and share |

The risk plot groups whole categories, so it gives a first, rough view of contamination risk. The task-by-task risk levels used for decontamination come later, from `clean_contamination/classify_tasks.py`, which also checks the wording of each intent and finds impossible tasks.