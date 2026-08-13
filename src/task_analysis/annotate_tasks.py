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
    if total <= 2:
        return "Easy"
    if total <= 5:
        return "Medium"
    return "Hard"
 

annotator = W.make_agent(ANNOTATOR_INSTRUCTIONS, TaskAnnotation, label="annotator")
 
DIMENSIONS = ["pages_to_traverse", "retrieval_type", "interaction", "target_locatability"]


#Template normalisation
 
def normalise_template(intent: str) -> str:
    t = intent
    t = re.sub(r'"[^"]*"', "X", t)
    t = re.sub(r"'[^']*'", "X", t)
    t = re.sub(r"\b[A-Z][a-z]+_[A-Z][a-z]+\d*\b", "USER", t)
    t = re.sub(r"\b[a-zA-Z]+\d+\b", "USER", t)
    t = re.sub(r"\$?\d[\d,.]*\b", "N", t)
    t = re.sub(r"\bsubreddit \w+", "subreddit X", t)
    t = re.sub(r"\bforum \w+", "forum X", t)
    t = re.sub(r"\br/\w+", "r/X", t)
    return re.sub(r"\s+", " ", t).strip().lower()



#Observed cost from a collection file
 
def load_observed_cost(episodes_path: str | None) -> dict:
    if not episodes_path or not Path(episodes_path).exists():
        return {}
 
    by_task = defaultdict(list)
    with open(episodes_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ep = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in ep:
                continue
            by_task[ep.get("task_id")].append(ep)
 
    observed = {}
    for tid, eps in by_task.items():
        acting = [e for e in eps if e.get("steps", 0) > 0]
        if not acting:
            continue
 
        urls = set()
        for e in acting:
            for s in e.get("step_log") or []:
                if s.get("url"):
                    urls.add(s["url"])
                if s.get("next_url"):
                    urls.add(s["next_url"])
 
        observed[tid] = {
            "median_steps": statistics.median(e["steps"] for e in acting),
            "max_steps": max(e["steps"] for e in acting),
            "median_tokens": statistics.median(e["total_tokens"] for e in acting),
            "distinct_urls": len(urls),
            "episodes": len(acting),
        }
    return observed

def format_observed(obs) -> str:
    if not obs:
        return ""
    return (
        f"\n Observed cost (a baseline single agent, across budget levels):\n"
        f"median steps taken: {obs['median_steps']:.0f}\n"
        f"maximum steps taken: {obs['max_steps']}\n"
        f"distinct pages visited: {obs['distinct_urls']}\n"
        f"median tokens: {obs['median_tokens']:.0f}\n"
        f"Use this to calibrate PAGES TO TRAVERSE only. The agent may have wandered, so distinct pages is an upper bound."
    )


#Annotation
 
def annotate(cfg: dict, category: str, observed) -> dict:
    intent = cfg.get("intent", "")
    eval_criteria = json.dumps(cfg.get("eval", {}))[:1200]
    sites = cfg.get("sites", [])
    site = sites[0] if sites else "unknown"
 
    prompt = (f"Site: {site}\n"
              f"Intent: {intent}\n"
              f"Evaluation criteria: {eval_criteria}"
              f"{format_observed(observed)}")
 
    a = W.call_agent(annotator, prompt).output
    total = sum(getattr(a, d) for d in DIMENSIONS)
 
    row = {
        "task_id": cfg.get("task_id"),
        "site": site,
        "intent": intent,
        "template": normalise_template(intent),
        "pages_to_traverse": a.pages_to_traverse,
        "retrieval_type": a.retrieval_type,
        "interaction": a.interaction,
        "target_locatability": a.target_locatability,
        "rubric_total": total,
        "difficulty_tier": tier_of(total),
        "task_category": category,
        "confidence": a.confidence,
        "justification": a.justification,
    }
    if observed:
        row["observed"] = observed
    return row


#Post-processing
 
def harmonise_templates(rows: list) -> int:
    groups = defaultdict(list)
    for r in rows:
        groups[r["template"]].append(r)
 
    adjusted = 0
    for tmpl, grp in groups.items():
        if len(grp) < 2:
            continue
        medians = {d: int(statistics.median(r[d] for r in grp))
                   for d in DIMENSIONS}
        for r in grp:
            changed = [d for d in DIMENSIONS if r[d] != medians[d]]
            if not changed:
                r["template_adjusted"] = False
                continue
            for d in changed:
                r[d] = medians[d]
            r["rubric_total"] = sum(r[d] for d in DIMENSIONS)
            r["difficulty_tier"] = tier_of(r["rubric_total"])
            r["template_adjusted"] = True
            r["adjusted_dimensions"] = changed
            adjusted += 1
    return adjusted


def flag_for_review(rows: list, threshold: float) -> None:
    for r in rows:
        reasons = []
 
        if r["confidence"] < threshold:
            reasons.append(f"low confidence ({r['confidence']:.2f})")
 
        if r.get("template_adjusted"):
            reasons.append(
                f"disagreed with template median on "
                f"{', '.join(r.get('adjusted_dimensions', []))}")
 
        #Score contradicts observed navigation
        obs = r.get("observed")
        if obs:
            pages = r["pages_to_traverse"]
            urls = obs["distinct_urls"]
            if pages == 0 and urls >= 3:
                reasons.append(f"scored 0 pages but agent visited {urls}")
            elif pages == 2 and urls <= 1:
                reasons.append(f"scored 2 pages but agent visited {urls}")
 
        r["needs_review"] = bool(reasons)
        r["review_reasons"] = reasons

