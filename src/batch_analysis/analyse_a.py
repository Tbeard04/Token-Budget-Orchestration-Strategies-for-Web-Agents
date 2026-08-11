from __future__ import annotations

import argparse
import statistics
from pathlib import Path

import pandas as pd

from analysis.shared import (
    load, join_tiers, print_section, run_shared_analysis,
    BUDGET_TERMINATIONS, STUCK_TERMINATIONS,
)


