"""
strategy_c.py - Strategy C: router-directed dynamic strategy.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import wa_env as W
import strategy_a
import strategy_b
from strategy_c.train.train_router import MLP, task_features, stop_features


DEFAULT_BUDGET = 16_000
#Path to the router-directed dynamic strategy model
ROUTER_DIR = Path("../data/processed/strategy_c_models/model_2")
#Path to the task metadata
TASK_METADATA = Path("../data/processed/final_annotation_difficulty_tiers/task_metadata.jsonl")
#Path to the task risk levels
TASK_RISK = Path("../data/processed/task_list/task_risk_levels.jsonl")
#Ordinal mapping for task difficulty tiers
TIER_ORD = {"Easy": 0, "Medium": 1, "Hard": 2}

#What a router-initiated stop submits. "none": end the episode with no answer
STOP_ANSWER = "none"

