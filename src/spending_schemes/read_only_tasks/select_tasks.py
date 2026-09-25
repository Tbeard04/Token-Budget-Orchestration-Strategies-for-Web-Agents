"""
select_tasks.py - pick the fixed task set
Read-only tasks only: every task is attempted 9 times (3 strategies x 3 spending schemes)
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

from spending_schemes import spending_scheme_config as C

METADATA = Path("../data/processed/final_annotation_difficulty_tiers/task_metadata.jsonl")
RISK = Path("../data/processed/task_list/task_risk_levels.jsonl")
OUT = Path("../data/processed/task_list/read_only_tasks.jsonl")