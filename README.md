## Requirements
```text
pydantic-ai>=1.0
browsergym
browsergym-webarena
playwright
openai
sentence-transformers
pandas
numpy
scipy
statsmodels
scikit-learn
xgboost 
matplotlib
seaborn
d3rlpy
python-dotenv
```

## Reproducing the Experiment & Environment
```text
This guide rebuilds the AWS environment used to collect every episode in this project (Strategies A, B and C, and the spending-scheme runs). It is reconstructed from the project journal. Stock WebArena will not run this project unmodified. The WebArena codebase (2023) is incompatible with the current OpenAI SDK and API, and BrowserGym's default action timeout is too short for this deployment. The required fixes are listed in Section 0 and applied in Sections 3–5.
```

### Section 0. Summary of deviations from stock WebArena/BrowserGym
---

| # | Component | Stock behaviour | This project |
|---|---|---|---|---|---|
| 1 | WebArena LLM evaluators (`llm_fuzzy_match`, `llm_ua_match`) | Call `gpt-4-1106-preview` | Call `gpt-4o-mini` |
| 2 | WebArena `llms` module | Imports `openai.error` (openai < 1.0) | `openai.error` stubbed at import time in `src/wa_env.py` | Module removed in openai ≥ 1.0; Pydantic AI requires openai 1.x, so downgrading was not an option |
| 3 | WebArena `generate_from_openai_chat_completion` | `openai.ChatCompletion.create` | openai ≥ 1.0 client, same signature, patched at runtime in `src/wa_env.py` |
| 4 | `HTMLContentEvaluator` in `~/webarena/evaluation_harness/evaluators.py` | Assumes `required_contents` is a string | Accepts string or dict |
| 5 | BrowserGym action timeout (`browsergym/core/action/functions.py`) | 500 ms per action | 10000 ms |
| 6 | Magento base URL | Set to the server hostname | `http://localhost:7770` / `:7780` as AWS Instance initialised with no Elastic IP |
| 7 | Sites hosted | 5 sites (+ map, wiki) | Scope on 3 sites: Shopping (7770), Shopping Admin/CMS (7780), Forum/Postmill (9999) |
---

## Directory Tree:
```text
token-budget-orchestration-strategies-for-web-agents/
│
├── src/
│   ├── wa_env.py           # shared WebArena/BrowserGym harness: task pools, agent construction, budget accounting
│   ├── run_batch.py        # batch driver: runs a strategy over N tasks x 6 budgets
│   ├── strategy_a.py       # Strategy A - single-agent baseline
│   ├── strategy_b.py       # Strategy B - fixed Planner -> Executor -> Critic pipeline
│   ├── .env                # API keys and site URLs (git-ignored)
│   │
│   ├── task_analysis/                  # what the tasks are
│   │   ├── extract_task_intent.py      # intents + regex task_category -> task_intents.json
│   │   ├── task_category_plots.py      # category distribution plots
│   │   └── test.raw.json               # Taken from WebArena repository
│   │
│   ├── task_annotation/                # how hard the tasks are (difficulty rubric)
│   │   ├── annotate_tasks.py           # entry point: gold set, annotation, re-band, merge
│   │   ├── rubric.py                   # the scale: 4 dimensions, prompt, band cut-points
│   │   ├── tasks.py                    # task pools, sampling, site allocation
│   │   ├── observed_cost.py            # per-task navigation cost from a collection file
│   │   ├── gold_set.py                 # blank gold set, exemplar split, merging hand labels
│   │   ├── annotator.py                # the LLM call and the output row
│   │   ├── postprocess.py              # template harmonisation, review flags, re-banding
│   │   ├── reporting.py                # terminal summaries and the validity check
│   │   ├── agreement.py                # Cohen's kappa, Landis & Koch bands, correlations
│   │   └── cohen_kappa.py              # entry point: human-vs-model agreement
│   │
│   ├── clean_contamination/            # decontamination pipeline (ran in this order)
│   │   ├── __init__.py
│   │   ├── remove_errors.py            # 1. strip timeout / environment errors
│   │   ├── sort_batch.py               # 2. sort episodes deterministically
│   │   ├── classify_tasks.py           # 3. read_only / idempotent / non_idempotent
│   │   ├── flag_episodes.py            # 4. per-strategy first-success rule
│   │   └── join_metadata.py            # 5. attach difficulty + category to episodes
│   │
│   └── batch_analysis/                 # results analysis
│       ├── __init__.py
│       ├── shared.py                   # loading, budget bucketing, plot helpers
│       ├── analyse_a.py                # Strategy A cost-performance curves
│       ├── analyse_b.py                # Strategy B, plus critic-invocation analysis
│       ├── compare_ab.py               # A vs B under matched budgets
│       └── compare_b_instructions.py   # B instruction versions v1 / v2 / v3
│
├── data/
│   ├── raw/                                    # collection output, never edited in place
│   │   ├── 6_budgets_ALL_tasks_batch/          # MAIN DATASET - all 475 single-site tasks
│   │   │   ├── batch_Strategy_A/
│   │   │   │   └── strategy_a.jsonl            # 2,850 episodes (475 tasks x 6 budgets)
│   │   │   └── batch_Strategy_B/
│   │   │       └── strategy_b.jsonl
│   │   │
│   │   ├── 6_budgets_201_tasks_batch/          # earlier 67-per-site sample
│   │   │   ├── 67_batch_Strategy_A/
│   │   │   │   ├── strategy_a.jsonl
│   │   │   │   └── analyse_batch_results_terminal_output.txt
│   │   │   └── 67_batch_Strategy_B/
│   │   │       ├── strategy_b.jsonl
│   │   │       └── analyse_batch_results_terminal_output.txt
│   │   │
│   │   ├── pilot_batch_outputs/                # first initial runs
│   │   │   ├── small_batch_test_Strategy_A/
│   │   │   │   ├── pilot_a.jsonl
│   │   │   │   ├── pilot_a_structured.json
│   │   │   │   └── Pilot test batch run for strategy A.txt
│   │   │   └── small_batch_test_Strategy_B/
│   │   │       ├── pilot_b.jsonl
│   │   │       ├── pilot_b_structured.json
│   │   │       └── Pilot test batch run for strategy B.txt
│   │   │
│   │   ├── annotation_tests/                   # difficulty-annotation runs
│   │   │   ├── run1/
│   │   │   │   ├── tasks_annotated_run1.jsonl
│   │   │   │   ├── tasks_annotated_structured_run1.json
│   │   │   │   └── readme.txt                  # with task_intents & A's output, no kappa
│   │   │   └── run2/
│   │   │       └── readme.txt                  # with task_intents & A's output + Cohen's kappa
│   │   │
│   │   ├── strategy_B_instruction_versions/    # prompt provenance for Strategy B
│   │   │   ├── v1_B_instructions.md
│   │   │   ├── v2_B_instructions.md
│   │   │   ├── v3_B_instructions.md
│   │   │   ├── strategy_b_v1_instructions.jsonl
│   │   │   ├── strategy_b_v2_instructions.jsonl
│   │   │   └── strategy_b_v3_instructions.jsonl
│   │   │
│   │   └── readme.txt
│   │
│   └── processed/                              # derived data, reproducible from raw/
│       ├── 6_budgets_ALL_tasks_decontaminated_batch/
│       │   ├── batch_strategy_A/
│       │   └── batch_strategy_B/
│       │
│       ├── task_list/
│       │   └── task_intents.json               # intents + task_category for all 475 tasks from task_analysis/
│       │
│       ├── final_annotation_difficulty_tiers/  # task_metadata.jsonl (final labels)
│       │
│       ├── diagrams/
│       │   ├── 6_budgets_ALL_tasks_decontaminated_batches/
│       │   │   ├── strategy_a_batch/
│       │   │   ├── strategy_b_batch/
│       │   │   └── comparisons/
│       │   ├── difficulty_tiers/
│       │   ├── pilot_C_batch/
│       │   ├── task_list/
│       │   │   ├── task_categories_overall.png
│       │   │   ├── task_categories_by_site.png
│       │   │   └── task_categories_risk.png
│       │   └── other_diagrams/
│       │       ├── 6_budgets_201_tasks/
│       │       │   ├── strategy_a_batch/
│       │       │   │   ├── cost_curve_a.png
│       │       │   │   ├── success_by_site_a.png
│       │       │   │   ├── termination_reasons_a.png
│       │       │   │   └── token_boxplot_a.png
│       │       │   └── strategy_b_batch/
│       │       │       ├── cost_curve_b.png
│       │       │       ├── success_by_site_b.png
│       │       │       ├── termination_reasons_b.png
│       │       │       └── token_boxplot_b.png
│       │       └── pilot_batch/
│       │           ├── pilot_batch_strategy A/
│       │           │   ├── cost_curve_a.png
│       │           │   ├── success_by_site_a.png
│       │           │   ├── termination_reasons_a.png
│       │           │   └── token_boxplot_a.png
│       │           └── pilot_batch_strategy B/
│       │               ├── cost_curve_b.png
│       │               ├── success_by_site_b.png
│       │               ├── termination_reasons_b.png
│       │               └── token_boxplot_b.png
│       │
│       └── readme.txt
│
├── docs/
│   ├── Additional Material/
│   └── Source Code.txt
│
├── keys/
│   └── webarena-key.pem               # AWS SSH key (git-ignored - never commit)
│
├── .venv/
├── .vscode/
├── requirements.txt
├── .gitignore
└── README.md
```
