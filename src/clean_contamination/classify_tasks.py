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

Run from src/: python -m clean_contamination.classify_tasks \ --annotations ../data/processed/task_list/task_metadata.jsonl \ --output ../data/processed/task_list/task_risk_levels.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


# Annotation categories that involve writing to the site
STATE_CHANGE_CATEGORIES = {"create", "modify_value", "delete", "bulk_action", "crud_operation", "form_fill",}

# Intent phrases indicating an idempotent write - repeating is harmless
IDEMPOTENT_KEYWORDS = ["add to wishlist", "wish list", "wishlist", "add to cart", "add to my cart", "subscribe", 
    "upvote", "up vote", "like all", "dislike all", "thumbs down", "thumbs up", "rate ", "star rating",]

# Intent phrases indicating the task only reads
ACTUALLY_READ_ONLY_KEYWORDS = ["show me", "tell me", "what is", "what are", "what do", "among the", "find the", "list out", "list the",
    "how many", "how much", "is there any", "compare", "count of", "which customer", "who has",]

