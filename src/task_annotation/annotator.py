"""
annotator.py - the LLM call, and the row it produces
"""
from __future__ import annotations

import json

import wa_env as W
from pathlib import Path
from observed_cost import format_observed
from rubric import (ANNOTATOR_INSTRUCTIONS, DIMENSIONS, TaskAnnotation, normalise_template, tier_of, total_of)


#function to build the annotator
def build_annotator(exemplar_block: str = ""):
    #Construct the scoring agent, calibrated on the exemplars if given
    return W.make_agent(ANNOTATOR_INSTRUCTIONS + exemplar_block, TaskAnnotation, label="annotator")

#function to annotate one task
def annotate_one(agent, cfg: dict, category: str, observed) -> dict:
    #JSON object keys are strings; readers must int() them back
    intent = cfg.get("intent", "")
    #convert the evaluation criteria to a string
    eval_criteria = json.dumps(cfg.get("eval", {}))[:1200]
    #get the sites
    sites = cfg.get("sites", [])
    site = sites[0] if sites else "unknown"

    prompt = (f"Site: {site}\n"
              f"Intent: {intent}\n"
              f"Evaluation criteria: {eval_criteria}"
              f"{format_observed(observed)}")

    #call the agent
    a = W.call_agent(agent, prompt).output
    #calculate the total score
    total = total_of(a)

    row = {
        "task_id": cfg.get("task_id"),
        "site": site,
        "intent": intent,
        "template": normalise_template(intent),
        **{d: getattr(a, d) for d in DIMENSIONS},
        "rubric_total": total,
        "difficulty_tier": tier_of(total),
        "task_category": category,
        "confidence": a.confidence,
    }
    if observed:
        row["observed"] = observed
    return row

#function to load the categories
def load_categories(intents_path: str | None) -> dict:
    #task_category per task from extract_task_intent.py output "task_categories" mapping
    if not intents_path or not Path(intents_path).exists():
        return {}
    data = json.load(open(intents_path))
    #get the task categories
    flat = data.get("task_categories") or {}
    if flat:
        #create a dictionary to store the categories
        cats = {int(k): v for k, v in flat.items()}
        #print the number of categories loaded
        print(f"[annotate] categories loaded for {len(cats)} tasks")
        return cats

    #create a dictionary to store the categories
    cats = {}
    #for each site data in the data
    for site_data in data.get("sites", {}).values():
        #for each category and info in the site data
        for cat, info in site_data.get("categories", {}).items():
            for t in info.get("tasks", []):
                cats[t["task_id"]] = cat
    total = data.get("summary", {}).get("total_tasks")
    print(f"[annotate] categories loaded for {len(cats)} tasks")
    if total and len(cats) < total:
        print(f"[annotate] WARNING: the intents file has no 'task_categories' "
              f"mapping, so only {len(cats)} of {total} tasks have a category. "
              f"Re-run extract_task_intent.py to add it.")
    return cats