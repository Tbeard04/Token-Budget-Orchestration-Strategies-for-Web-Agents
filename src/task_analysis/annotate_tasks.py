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
    pages_to_traverse: int = Field(ge=0, le=2,
        description="0 single page, 1 two-to-three pages, 2 four-plus or unbounded")
    retrieval_type: int = Field(ge=0, le=2,
        description="0 read one value, 1 compare or filter a few, "
                    "2 aggregate or count over a set")
    interaction: int = Field(ge=0, le=2,
        description="0 read-only, 1 one form or click sequence, "
                    "2 multi-step state change")
    target_locatability: int = Field(ge=0, le=2,
        description="0 target named explicitly, 1 derivable from the page, "
                    "2 must be discovered by scanning")
    confidence: float = Field(ge=0.0, le=1.0,
        description="Honest confidence. Use below 0.7 when the task text is "
                    "ambiguous about how much navigation it requires.")
    justification: str = Field(
        description="One sentence explaining the scores")

