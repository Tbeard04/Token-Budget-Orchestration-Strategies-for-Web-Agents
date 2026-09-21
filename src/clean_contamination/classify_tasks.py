"""
classify_tasks.py - assign a contamination risk level to every task.

    read_only: Information retrieval, navigation, browsing. The agent reads but never writes. Running the same task again encounters an identical site, so every episode is clean no matter what ran before.

    ------------------------------------------------------------------------------------------------
    idempotent: Add to wishlist, add to cart, upvote, subscribe, rate. These DO change site state, 
    but repeating them does not change the outcome: an item already in the wishlist stays in the wishlist, an already
    upvoted post stays upvoted. The evaluator sees the correct final state whether the agent acted once or twice. Treated as clean.

    ------------------------------------------------------------------------------------------------
    non_idempotent: Change a price, delete reviews, post new content, update an address, cancel an order. A prior success leaves the site in a different
    starting condition, so a later episode may "succeed" by finding the work already done. Only these need contamination handling.

src/: python -m clean_contamination.classify_tasks \ --annotations ../data/processed/final_annotation_difficulty_tiers/task_metadata.jsonl \ --output ../data/processed/task_list/task_risk_levels.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

#risk levels
RISK_LEVELS = ["read_only", "idempotent", "non_idempotent"]

#Annotation categories that involve writing to the site
STATE_CHANGE_CATEGORIES = {"create", "modify_value", "delete", "bulk_action", "crud_operation", "form_fill",}

#Intent phrases indicating an idempotent write - repeating is harmless
IDEMPOTENT_KEYWORDS = ["add to wishlist", "wish list", "wishlist", "add to cart", "add to my cart", "to the shopping cart", "to my cart", "subscribe", "upvote", "up vote", "downvote", "down vote", "thumbs up", "thumbs down",]

#Intent phrases indicating the task only reads
ACTUALLY_READ_ONLY_KEYWORDS = ["show me", "tell me", "what is", "what are", "what do", "among the", "find the", "list out", "list the", "how many", "how much", "is there any", "compare", "count of", "which customer", "who has",]

#Patterns for writing verbs
WRITE_VERB_PATTERNS = [
    r"\breorder\b", r"\bre-order\b",
    r"\bbuy\b", r"\bpurchase\b", r"\bcheckout\b", r"\bplace an order\b",
    r"\bdelete\b", r"\bremove\b", r"\bcancel\b",
    r"\bchange\b", r"\bupdate\b", r"\bmodify\b", r"\bedit\b", r"\brename\b",
    r"\bset the\b", r"\bincrease\b", r"\breduce\b", r"\bdecrease\b",
    r"\bdisable\b", r"\benable\b", r"\bapprove\b",
    r"\bpost\b", r"\bcreate\b", r"\bdraft\b", r"\bsubmit\b", r"\breply\b",
    r"\bsubscribe\b", r"\bupvote\b", r"\bdownvote\b",
]

#check if the intent contains a phrase
def _has_phrase(intent: str, phrases: list[str]) -> bool:
    return any(p in intent for p in phrases)
 
#check if the intent contains a write verb
def _has_pattern(intent: str, patterns: list[str]) -> bool:
    return any(re.search(p, intent) for p in patterns)

def classify(intent: str, category: str) -> tuple[str, str]:
    #convert intent to lowercase
    i = (intent or "").lower()
    #if the category is a state change category
    if category in STATE_CHANGE_CATEGORIES:
        if _has_phrase(i, IDEMPOTENT_KEYWORDS):
            return "idempotent", f"{category}: idempotent action"
        return "non_idempotent", f"{category}: writes to the site"
    #if the intent contains a write verb
    if _has_pattern(i, WRITE_VERB_PATTERNS):
        if _has_phrase(i, IDEMPOTENT_KEYWORDS):
            return "idempotent", f"{category}: write verb, idempotent action"
        return "non_idempotent", f"{category}: write verb in the intent"
    #if the intent contains a read-only phrasing
    if _has_phrase(i, ACTUALLY_READ_ONLY_KEYWORDS):
        return "read_only", f"{category}: read-only phrasing"
    #if no write is detected
    return "read_only", f"{category}: no write detected"



# def main(annotations: Path, output: Path) -> None:
#     with open(annotations, "r") as f:
#         tasks = [json.loads(line) for line in f]
#     with open(output, "w") as f: