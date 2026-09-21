"""
flag_episodes.py - decide which episodes Strategy C may learn a reward from.
"""
from __future__ import annotations
 
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
 
RISK_LEVELS = ["read_only", "idempotent", "non_idempotent"]
 
 
def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows