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

# Structured output - the model is constrained to return these four rubric
# scores (0-2 each) plus derived labels. Storing the individual dimensions,
# not just the final tier, means the bands can be adjusted later without
# re-annotating: just re-band from the stored dimension scores.
# ---------------------------------------------------------------------------
class TaskAnnotation(BaseModel):
    pages_to_traverse: int = Field(
        ge=0, le=2,
        description="0 single page, 1 two-to-three pages, 2 four-plus or unbounded",
    )
    retrieval_type: int = Field(
        ge=0, le=2,
        description="0 read one value, 1 compare or filter a few, "
                    "2 aggregate or count over a set",
    )
    interaction: int = Field(
        ge=0, le=2,
        description="0 read-only, 1 one form or click sequence, "
                    "2 multi-step state change",
    )
    target_locatability: int = Field(
        ge=0, le=2,
        description="0 target named explicitly in the task, "
                    "1 derivable from the page, 2 must be discovered by scanning",
    )
    task_category: TaskCategory
    interaction_types: list[str] = Field(
        description="UI interactions involved, e.g. ['search', 'click', 'form_fill']"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Honest confidence in this classification",
    )
    justification: str = Field(
        description="One sentence explaining the scores"
    )


ANNOTATOR_INSTRUCTIONS = """\
You classify WebArena web-agent tasks for a study of token budgets. You are
given a task's natural-language intent, its evaluation criteria, and the site
it runs on. Score it on four dimensions, each 0-2.

PAGES TO TRAVERSE - how many distinct pages must be visited?
  0 = everything needed is on the starting page
  1 = two or three pages
  2 = four or more, or an unbounded search across pages

RETRIEVAL TYPE - what must be done with the information?
  0 = read a single stated value
  1 = compare or filter a small number of items
  2 = aggregate, count, or reason over a set of items

INTERACTION - what must be done to the site?
  0 = read-only; nothing on the site changes
  1 = one form submission or click sequence
  2 = a multi-step state change (create, edit, delete, configure)

TARGET LOCATABILITY - how hard is the target to find?
  0 = named explicitly in the task, e.g. "the Sprite Stasis Ball"
  1 = derivable from what is on the page
  2 = must be discovered by scanning or searching

Score from the task description alone. Do NOT guess how well an agent would
perform - you are measuring the task's structural demands, not an agent's
capability. Judge on the number of distinct interactions and the reasoning
depth required, not on surface wording or sentence length.

Also assign one task_category, list the UI interaction types involved, give
an honest confidence between 0 and 1, and justify the scores in one sentence.
"""

def tier_of(total: int) -> str:
    """Band a rubric total (0-8) into Easy/Medium/Hard.

    Cut-points are 2 and 5. Adjust here if the initial distribution comes
    out lopsided; because dimension scores are stored per task, re-banding
    is a one-liner over the JSONL and does not need re-annotation.
    """
    if total <= 2:
        return "Easy"
    if total <= 5:
        return "Medium"
    return "Hard"

annotator = W.make_agent(ANNOTATOR_INSTRUCTIONS, TaskAnnotation, label="annotator")


# Sampling: MUST match run_batch.py exactly, or the tier labels will not
# cover the same tasks the collection ran on.
# ---------------------------------------------------------------------------
def sample_tasks(n: int, sites: list[str], seed: int) -> list[int]:
    pools = W.single_site_tasks(sites)
    rng = random.Random(seed)
    ids: list[int] = []
    for site in sites:
        pool = pools.get(site, [])
        if pool:
            ids.extend(rng.sample(pool, min(n, len(pool))))
    return sorted(ids)
