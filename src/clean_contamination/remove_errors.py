"""
This script is used to remove errors from the data (not action errors)
"""

from __future__ import annotations
 
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
 
 
def classify_error(msg: str) -> str:
    """Group an error message into a readable category."""
    m = (msg or "").lower()
    if "timeout" in m and "goto" in m:
        return "page load timeout"
    if "target crashed" in m:
        return "Chromium crash"
    if "connection_refused" in m or "connection refused" in m:
        return "container not running"
    if "does not exist" in m or "notfounderror" in m:
        return "model/API not found"
    if "timeout" in m:
        return "timeout (other)"
    if "keyboardinterrupt" in m:
        return "interrupted"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--backup", default=None, help="backup path (default: <file>.bak)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--dry-run", action="store_true", help="report what would be removed, change nothing")
    args = ap.parse_args()
 
    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

     # Read and separate
    keep_lines: list[str] = []
    error_rows: list[dict] = []
    malformed = 0
 
    with path.open() as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rec = json.loads(stripped)
            except json.JSONDecodeError:
                malformed += 1
                continue
 
            # Episode-level "error" key = the episode never completed.
            # An action_error inside step_log is NOT this.
            if "error" in rec:
                error_rows.append(rec)
            else:
                keep_lines.append(line if line.endswith("\n") else line + "\n")
 
    total = len(keep_lines) + len(error_rows)
 
    print(f"File: {path}")
    print(f"Episodes: {total}")
    print(f"valid: {len(keep_lines)}")
    print(f"errors: {len(error_rows)}")
    if malformed:
        print(f"malformed lines skipped: {malformed}")
 
    if not error_rows:
        print("\nNo environment errors found. Nothing to do.")
        return
 
    # Show exactly what will be removed
    print(f"\n{'=' * 70}\nEpisodes that will be removed\n{'=' * 70}")
 
    categories = Counter(classify_error(r.get("error", "")) for r in error_rows)
    print("\nBy cause:")
    for cat, n in categories.most_common():
        print(f"   {cat:26s} {n:>4}")
 
    by_strategy = Counter(r.get("strategy", "?") for r in error_rows)
    by_budget = Counter(r.get("budget_level", "?") for r in error_rows)
    print("\nBy strategy: " + "  ".join(f"{k}:{v}" for k, v in sorted(by_strategy.items())))
    print("By budget  : " + "  ".join(f"{k}:{v}" for k, v in sorted(by_budget.items(), key=lambda x: str(x[0]))))
 
    print(f"\nFull list:")
    for r in error_rows:
        msg = str(r.get("error", "")).replace("\n", " ")[:64]
        print(f" task {str(r.get('task_id','?')):>5} @ "
              f"{str(r.get('budget_level','?')):>6}  "
              f"[{r.get('strategy','?')}] {r.get('site','?'):16s} {msg}")