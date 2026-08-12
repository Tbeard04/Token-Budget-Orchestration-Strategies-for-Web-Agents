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

