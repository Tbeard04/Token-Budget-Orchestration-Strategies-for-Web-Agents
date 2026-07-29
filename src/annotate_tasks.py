"""
annotate_tasks.py - derive a difficulty tier and category for each task.

WebArena ships no difficulty labels. RQ2 in this study asks how the three
strategies differ in sensitivity to task difficulty, so those labels must be
derived here. Two properties make them defensible:

1. Scored from the TASK TEXT ONLY, never from episode outcomes. If difficulty
   were defined by "Strategy A failed", then "harder tasks have lower success"
   would be true by construction and useless as a finding. The rubric is
   applied before, and independently of, any results.

2. Scored on four CONCRETE dimensions rather than a holistic judgement. This
   makes the LLM far more consistent across tasks and the instrument far
   easier to justify in the methodology.


THE RUBRIC (each dimension scored 0-2, total 0-8)
==================================================

  PAGES TO TRAVERSE - how many distinct pages must be visited?
    0  single page             everything needed is on the starting page
    1  two or three pages      moderate navigation
    2  four or more pages      unbounded search, or deep hierarchy

  RETRIEVAL TYPE - what must be done with the information found?
    0  read one value          a single stated fact
    1  compare or filter       a small number of items
    2  aggregate over a set    count, sum, or reason over many items

  INTERACTION - what changes on the site?
    0  read-only               nothing changes; pure information task
    1  one form or click       a single state change
    2  multi-step state change create, edit, delete, configure a resource

  TARGET LOCATABILITY - how hard is the target to find?
    0  named explicitly        e.g. "the Sprite Stasis Ball"
    1  derivable from context  can be worked out from what is on the page
    2  must be discovered      requires searching or scanning

BAND TOTALS TO TIERS
  0-2  Easy      simple, single-step lookups
  3-5  Medium    moderate navigation or reasoning
  6-8  Hard      multi-hop traversal plus aggregation or state change


Validation (before relying on the output)
-----------------------------------------
  a) Hand-label roughly 50 tasks, then run the annotator over the same tasks and report Cohen's kappa.

  b) After collection, check the tiers correlate with observed median tokens
     and step counts. Correlation with COST (not success) is the honest test:
     success is the thing being explained, so validating against it would be
     circular.


Run
---
    # annotate exactly the tasks run_batch samples (same seed, same logic)
    python annotate_tasks.py --n 67

    # or a specific list, e.g. a hand-labelled gold set
    python annotate_tasks.py --tasks 47 276 623 109
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

import wa_env as W

# Task categories - six buckets that cover the observed range of WebArena tasks on the three target sites.
# ---------------------------------------------------------------------------
TaskCategory = Literal[
    "navigation",       # reach a target page or UI state
    "single_query",     # retrieve a fact from one page
    "multi_hop_query",  # retrieve a fact requiring traversal across pages
    "form_fill",        # complete and submit a form
    "crud_operation",   # create / update / delete a resource
    "composite",        # two or more of the above in sequence
]

