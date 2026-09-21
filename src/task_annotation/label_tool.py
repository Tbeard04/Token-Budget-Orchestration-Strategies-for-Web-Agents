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


# Making the evaluation criteria readable
def summarise_eval(raw: str) -> tuple[str, str]:
    #Return (eval_type, target) from the stored criteria JSON.
    if not raw:
        return "", ""
    try:
        e = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r'"eval_types":\s*\[([^\]]*)\]', raw)
        types = m.group(1).replace('"', "").replace(" ", "") if m else "?"
        return f"{types} (truncated)", raw[:120]

    types = "+".join(e.get("eval_types") or [])

    ans = e.get("reference_answers")
    if ans:
        for key in ("exact_match", "must_include", "fuzzy_match"):
            if key in ans:
                v = ans[key]
                v = ", ".join(str(x) for x in v) if isinstance(v, list) else str(v)
                return types, f"{key}: {v[:200]}"

    if e.get("reference_url"):
        return types, f"url: {e['reference_url'][:200]}"

    ph = e.get("program_html") or []
    if ph:
        urls = {p.get("url", "") for p in ph}
        return types, f"DOM check x{len(ph)} on {len(urls)} page(s)"

    return types, ""

# JSONL <--> CSV conversion
def load_gold(path: str) -> list[dict]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows