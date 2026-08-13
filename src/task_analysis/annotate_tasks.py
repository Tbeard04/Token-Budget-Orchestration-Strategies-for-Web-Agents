from __future__ import annotations
 
import argparse
import json
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
 
from pydantic import BaseModel, Field
 
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import wa_env as W

#Rubric output
#The LLM scores four dimensions and nothing else. Category comes from extract_task_intents.py, which is regex-based, self-tested and consistent.
 
class TaskAnnotation(BaseModel):
    pages_to_traverse: int = Field(ge=0, le=2, description="0 single page, 1 two-to-three pages, 2 four-plus or unbounded")
    retrieval_type: int = Field(ge=0, le=2, description="0 read one value, 1 compare or filter a few, 2 aggregate or count over a set")
    interaction: int = Field(ge=0, le=2, description="0 read-only, 1 one form or click sequence, 2 multi-step state change")
    target_locatability: int = Field(ge=0, le=2, description="0 target named explicitly, 1 derivable from the page, 2 must be discovered by scanning")
    confidence: float = Field(ge=0.0, le=1.0, description="Honest confidence. Use below 0.7 when the task text is ambiguous about how much navigation it requires.")
    justification: str = Field(description="One sentence explaining the scores")


ANNOTATOR_INSTRUCTIONS = """\
You score WebArena web-agent tasks on four difficulty dimensions for a study
of token budgets.
 
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
 
You measure the task's STRUCTURAL DEMANDS, not an agent's capability. Judge on
the number of distinct interactions and the reasoning depth required, not on
sentence length or surface wording.
 
You may also be shown OBSERVED COST: how many steps and distinct pages a
baseline agent actually needed. Use it to calibrate PAGES TO TRAVERSE, which
is difficult to judge from text alone. Do NOT use it to infer the other three
dimensions, and note that an agent may wander - observed pages is an upper
bound on what the task requires, not the exact number.
 
Give an honest confidence. Use below 0.7 when the task text leaves the amount
of navigation genuinely ambiguous.
"""

def tier_of(total: int) -> str:
    """Band a rubric total (0-8). Dimension scores are stored separately so
    the bands can be changed later with --reband, no re-annotation needed."""
    if total <= 2:
        return "Easy"
    if total <= 5:
        return "Medium"
    return "Hard"
 

annotator = W.make_agent(ANNOTATOR_INSTRUCTIONS, TaskAnnotation, label="annotator")
 
DIMENSIONS = ["pages_to_traverse", "retrieval_type", "interaction", "target_locatability"]