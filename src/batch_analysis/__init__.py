"""
Modules:
    shared      functions used by both strategy A and B
    analyse_a   Strategy A (single-agent baseline)
    analyse_b   Strategy B (fixed Planner->Executor->Critic pipeline)
    compare_ab  A vs B comparison and Strategy C design inputs

Run from src/ as modules so the package imports resolve:
    python -m analysis.analyse_a  --file ../data/raw/strategy_a.jsonl
    python -m analysis.analyse_b  --file ../data/raw/strategy_b.jsonl
    python -m analysis.compare_ab --a ../data/raw/strategy_a.jsonl \
                                  --b ../data/raw/strategy_b.jsonl
"""
