"""
This script is used to remove environment errors from the data (not action errors)
"""

from __future__ import annotations
 
import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

#Group an error message into a readable category
#function to classify the error
def classify_error(msg: str) -> str:
    m = (msg or "").lower()
    #if the message contains timeout and goto
    if "timeout" in m and "goto" in m:
        #return the page load timeout
        return "page load timeout"
    #if the message contains target crashed
    if "target crashed" in m:
        #return the Chromium crash
        return "Chromium crash"
    if "connection_refused" in m or "connection refused" in m:
        #return the container not running
        return "container not running"
    #if the message contains does not exist or notfounderror
    if "does not exist" in m or "notfounderror" in m:
        #return the model/API not found
        return "model/API not found"
    #if the message contains timeout
    if "timeout" in m:
        return "timeout (other)"
    #if the message contains keyboardinterrupt
    if "keyboardinterrupt" in m:
        #return the interrupted
        return "interrupted"
    #return other
    return "other"

#function to main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True)
    ap.add_argument("--backup", default=None, help="backup path (default: <file>.bak)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    path = Path(args.file)
    if not path.exists():
        raise SystemExit(f"File not found: {path}")

    #Read and separate
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

    #function to show the exactly what will be removed
    categories = Counter(classify_error(r.get("error", "")) for r in error_rows)
    #print the by cause
    print("\nBy cause:")
    for cat, n in categories.most_common():
        print(f" {cat:26s} {n:>4}")

    #function to show the by strategy
    by_strategy = Counter(r.get("strategy", "?") for r in error_rows)
    #function to show the by budget
    by_budget = Counter(r.get("budget_level", "?") for r in error_rows)
    print("\nBy strategy: " + "  ".join(f"{k}:{v}" for k, v in sorted(by_strategy.items())))
    print("By budget: " + "  ".join(f"{k}:{v}" for k, v in sorted(by_budget.items(), key=lambda x: str(x[0]))))

    print(f"\nFull list:")
    for r in error_rows:
        msg = str(r.get("error", "")).replace("\n", " ")[:64]
        print(f" task {str(r.get('task_id','?')):>5} @ "
              f"{str(r.get('budget_level','?')):>6}  "
              f"[{r.get('strategy','?')}] {r.get('site','?'):16s} {msg}")

    #Safety check: these rows must not contain real episode data
    suspicious = [r for r in error_rows if r.get("step_log") or r.get("total_tokens")]
    if suspicious:
        print(f"\n!! WARNING: {len(suspicious)} rows have an 'error' key but " f"also contain step_log or token data.")
        print("Inspection:")
        for r in suspicious[:5]:
            print(f"task {r.get('task_id')} @ {r.get('budget_level')}")

    #Confirm
    if not args.yes:
        reply = input("\nProceed? [y/N] ").strip().lower()
        if reply not in ("y", "yes"):
            print("Aborted. No changes made.")
            return

    #Back up, then write
    backup = Path(args.backup) if args.backup else path.with_suffix(
        path.suffix + ".bak")
    shutil.copy2(path, backup)
    print(f"\nBackup written to {backup}")

    with path.open("w") as f:
        f.writelines(keep_lines)

    print(f"Removed {len(error_rows)} error episodes")
    print(f"{path} now contains {len(keep_lines)} episodes")

if __name__ == "__main__":
    main()