"""
rubric.py - the difficulty instrument itself. Contains the four dimensions, the scoring guide given to the annotator, the band cut-points, and the template normaliser.
"""
from __future__ import annotations

import re

from pydantic import BaseModel, Field

#define the dimensions
DIMENSIONS = ["pages_to_traverse", "retrieval_type", "interaction", "target_locatability"]

#define the tiers
TIERS = ["Easy", "Medium", "Hard"]

#define the easy max
EASY_MAX = 2
#define the medium max
MEDIUM_MAX = 5

#define the task annotation model
class TaskAnnotation(BaseModel):
    pages_to_traverse: int = Field(ge=0, le=2, description="0 single page, 1 two-to-three pages, 2 four-plus or unbounded")
    retrieval_type: int = Field(ge=0, le=2, description="0 read one value, 1 compare or filter a few, 2 aggregate or count over a set")
    interaction: int = Field(ge=0, le=2, description="0 read-only, 1 one form or click sequence, 2 multi-step state change")
    target_locatability: int = Field(ge=0, le=2, description="0 target named explicitly, 1 derivable from the page, 2 must be discovered by scanning")
    confidence: float = Field(ge=0.0, le=1.0, description="Honest confidence. Use below 0.7 when the task text is ambiguous about how much navigation it requires.")

#define the annotator instructions
ANNOTATOR_INSTRUCTIONS = """\
You score WebArena web-agent tasks on four difficulty dimensions for a study of token budgets.

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

#define the rubric summary
RUBRIC_SUMMARY = """\
  pages_to_traverse    0 one page        1 two-three      2 four+/unbounded
  retrieval_type       0 one value       1 compare/filter 2 aggregate/count
  interaction          0 read-only       1 one form/click 2 multi-step change
  target_locatability  0 named in task   1 derivable      2 must be discovered"""


#Band a rubric total (0-8) into Easy/Medium/Hard
def tier_of(total: int) -> str:
    if total <= EASY_MAX:
        return "Easy"
    if total <= MEDIUM_MAX:
        return "Medium"
    return "Hard"

# Sum the four dimension scores of a dict or an annotation object
def total_of(row) -> int:
    if isinstance(row, dict):
        return sum(int(row[d]) for d in DIMENSIONS)
    return sum(int(getattr(row, d)) for d in DIMENSIONS)

#function to normalise the template
def normalise_template(intent: str) -> str:
    #normalise the template
    t = intent
    #replace the quoted strings with X
    t = re.sub(r'"[^"]*"', "X", t)
    #replace the single quoted strings with X
    t = re.sub(r"'[^']*'", "X", t)
    #replace the user strings with USER
    t = re.sub(r"\b[A-Z][a-z]+_[A-Z][a-z]+\d*\b", "USER", t)
    t = re.sub(r"\b[a-zA-Z]+\d+\b", "USER", t)
    #replace the numbers with N
    t = re.sub(r"\$?\d[\d,.]*\b", "N", t)
    #replace the subreddit strings with subreddit X
    t = re.sub(r"\bsubreddit \w+", "subreddit X", t)
    #replace the forum strings with forum X
    t = re.sub(r"\bforum \w+", "forum X", t)
    #replace the r/ strings with r/X
    t = re.sub(r"\br/\w+", "r/X", t)
    #replace the whitespace with a single space
    return re.sub(r"\s+", " ", t).strip().lower()