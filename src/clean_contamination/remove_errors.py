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