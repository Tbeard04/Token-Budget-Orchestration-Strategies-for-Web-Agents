"""
label_tool.py - hand-label the gold set in a spreadsheet instead of raw JSONL so its easier to work with.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path

from rubric import DIMENSIONS, RUBRIC_SUMMARY, TIERS, normalise_template, tier_of

COLUMNS = [
    "task_id", "site", "group", "intent", "eval_type", "eval_target",
    "obs_median_steps", "obs_max_steps", "obs_urls",
    "pages_to_traverse", "retrieval_type", "interaction", "target_locatability",
]
