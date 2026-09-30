#test
## add spending schemes analysis here for shared analysis between A,B & C
## 1. pay-as-you-go which is the full batches completed already 
## IMPORTANT NOTE: READ-ONLY TASKS (100 tasks) SO THEY ARE NOT CONTAMINATED BY MODIFIED STATE TASKS
## 2. front loaded = the first step can use up to 40% of the budget, and the later steps share the rest
## 3. Even scheme = the budget is split equally over the number of steps that strategy usually takes at 64k
## 4. Reactive scheme = each step starts with the even share, gets 1.5× after a failed move or when the page didn't change, and 0.75× after a move that made progress
#test

from __future__ import annotations
 
import argparse
import itertools
import json
from pathlib import Path
 
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 
from batch_analysis.shared import (load, print_section, wilson, mcnemar_exact, outcome_group, COLOURS, MARKERS, NAMES, TIER_ORDER, OUTCOME_ORDER, OUTCOME_COLOURS)

BUDGET = 32_000
STRATEGIES = ["A", "B", "C"]
SCHEMES = ["pay_as_you_go", "even", "front_loaded", "reactive"]
SCHEME_LABELS = {"pay_as_you_go": "Pay as you go", "even": "Even", "front_loaded": "Front-loaded", "reactive": "Reactive"}