Directory tree for additional material

```text
Additional Material/
├── README.md                             # this file
├── AI Usage.md                           # the generative-AI usage log behind report Chapter 5 / Appendix D
├── Journal.docx                          # daily project journal (submission only - not in the public repository)
├── Source Code.txt                       # full source code in txt file
├── Proposal Token..docx                  # final preliminary proposal
├── Token Budget Orchestration..docx      # final report in docx format
├── Token-Budget-Orchestration..zip       # full source code compressed to zip
|
├── Datasets/                             # copies of the key datasets (originals under data/ in the repo)
│   ├── Annotated Task Difficulty Tiers/  # gold set + final difficulty labels
│   │   ├── gold_set_50.csv
│   │   └── task_metadata.jsonl
|   |
│   ├── Batch Outputs/                    # the decontaminated final_episodes files
│   │   ├── final_episodes.jsonl
│   │   ├── final_episodes_a.jsonl
│   │   ├── final_episodes_b.jsonl
│   │   └── final_episodes_c.jsonl
|   |
│   ├── Router C Training Dataset/        # mode_pairs + transitions
│   │   ├── mode_pairs.jsonl
│   │   └── transitions.jsonl
|   |
│   ├── Scheme Output/                    # the 900 RQ3 episodes
│   │   └── schemes.jsonl
|   |
│   └── Task Lists/                       # task configs, intents, risk levels, RQ3 sample
│       ├── read_only_tasks.jsonl
│       ├── task_intents.json
│       ├── task_risk_levels.jsonl
│       └── test.raw.json
|
├── Diagrams/                             # every figure used in the report, grouped by chapter topic
│   ├── Batch Outputs/                    # main-grid results figures (Ch.4)
│   │   ├── Comparisons/                  # A vs B vs C figures
│   │   │   ├── cost_curve_a_vs_b_vs_c.png
│   │   │   ├── cost_performance_a_vs_b_vs_c.png
│   │   │   ├── difficulty_a_vs_b_vs_c.png
│   │   │   ├── difficulty_easy_by_budget_a_vs_b_vs_c.png
│   │   │   ├── difficulty_hard_by_budget_a_vs_b_vs_c.png
│   │   │   ├── difficulty_medium_by_budget_a_vs_b_vs_c.png
│   │   │   ├── difficulty_sensitivity_a_vs_b_vs_c.png
│   │   │   ├── outcome_mix_a_vs_b_vs_c.png
│   │   │   ├── outcome_mix_by_tier_a_vs_b_vs_c.png
│   │   │   ├── per_step_cost_a_vs_b_vs_c.png
│   │   │   └── token_efficiency_a_vs_b_vs_c.png
|   |   |
│   │   ├── Strategy A/
│   │   │   ├── cost_curve_a.png
│   │   │   ├── difficulty_curve_a.png
│   │   │   ├── success_by_site_a.png
│   │   │   ├── termination_reasons_a.png
│   │   │   └── token_boxplot_a.png
|   |   |
│   │   ├── Strategy B/
│   │   │   ├── cost_curve_b.png
│   │   │   ├── critic_analysis_b.png
│   │   │   ├── difficulty_curve_b.png
│   │   │   ├── role_cost_breakdown_b.png
│   │   │   ├── success_by_site_b.png
│   │   │   ├── termination_reasons_b.png
│   │   │   └── token_boxplot_b.png
|   |   |
│   │   └── Strategy C/
│   │       ├── cost_curve_c.png
│   │       ├── difficulty_curve_c.png
│   │       ├── router_stop_by_tier_c.png
│   │       ├── router_stop_rate_c.png
│   │       ├── router_stop_step_c.png
│   │       ├── success_all_vs_attempted_c.png
│   │       ├── success_by_site_c.png
│   │       ├── termination_reasons_c.png
│   │       ├── tier_all_vs_attempted_c.png
│   │       └── token_boxplot_c.png
|   |
│   ├── Difficulty Tiers/                 # tier composition figures (Ch.3)
│   │   ├── tier_by_category.png
│   │   └── tier_by_site.png
|   |
│   ├── Drawn Diagrams/                   # hand-drawn figures: work plan, control flow, Strategy C, decontamination
│   │   ├── Decontamination Pipeline.png
│   │   ├── Step control flow of strategies.png
│   │   ├── Strategy C Offline training & live use.png
│   │   └── Work Plan.png
|   |
│   ├── Router C Models/                  # router training/selection figures
│   │   ├── frontier.png
│   │   ├── per_budget_tokens.png
│   │   └── stop_step0.png
|   |
│   ├── Scheme Outputs/                   # RQ3 figures
│   │   ├── outcome_mix.png
│   │   ├── tokens_by_scheme.png
│   │   └── tokens_per_success.png
|   |
│   └── Task List/                        # task category figures
│       ├── task_categories_by_site.png
│       └── task_categories_overall.png
|
└── Research/                             # the cited papers, grouped by literature (Ch.2)
    ├── 1 Benchmark and Harness/
    │   ├── LeSellierDeChezelles2024_BrowserGym.pdf
    │   ├── Xu2024_TheAgentCompany.pdf
    │   └── Zhou2023_WebArena.pdf
    ├── 2 Multi-agent orchestration/
    │   ├── Wang2024_MixtureOfAgents.pdf
    │   └── YangB2025_CodeAgents.pdf
    ├── 3 Token cost and budget aware systems/
    │   ├── Cai2025_AgentBalance.pdf
    │   ├── Chen2026_TokenEconomics.pdf
    │   ├── Li2026_BAVT_SpendLessReasonBetter.pdf
    │   ├── Salim2026_Tokenomics.pdf
    │   ├── YangL2025_BAMAS.pdf
    │   └── Zhang2024_CutTheCrap.pdf
    ├── 4 Evaluation Methodology/
    │   ├── Mohammadi2025_EvaluationBenchmarkingLLMAgents.pdf
    │   └── WangJain2024_ReasoningInTokenEconomies.pdf
    ├── 5 Cohen Kappa/
    │   ├── Cohen, J Psychological Bulletin.pdf
    │   └── Landis-MeasurementObserverAgreement-1977.pdf
    └── 6 Other/
        ├── Dunn-MultipleComparisonsAmong-1961.pdf
        ├── Shi2017_WorldOfBits.pdf
        ├── Spearman1904_ProofAndMeasurement.pdf
        ├── Wilcoxon-IndividualComparisonsRanking-1945.pdf
        └── Wilson-ProbableInferenceLaw-1927.pdf
```