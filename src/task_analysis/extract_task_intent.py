"""
extract_task_intents.py - extract task IDs and intents for the three target sites, grouped by site and by task type. Produces a readable JSON file for reviewing what the agent prompts need to cover.

Run:
    python extract_task_intents.py --input test.raw.json --output ../data/task_list/task_intents.json
"""
import json
import argparse
from collections import defaultdict

TARGET_SITES = {"shopping", "shopping_admin", "reddit"}


def categorise(intent: str) -> str:
    i = intent.lower()
    if any(w in i for w in ["what is", "tell me", "how much", "how many",
                             "list out", "presents", "get the", "count of",
                             "show me the email", "show me the name",
                             "show me the customer", "which customer",
                             "what are", "what do", "find the customer"]):
        return "information_retrieval"
    if any(w in i for w in ["like all", "dislike all", "thumbs down"]):
        return "bulk_action"
    if any(w in i for w in ["delete", "remove"]):
        return "delete"
    if any(w in i for w in ["reduce", "increase", "change", "update",
                             "modify", "set", "mark all", "disable",
                             "cancel order", "approve"]):
        return "modify_value"
    if any(w in i for w in ["search", "find", "browse", "look", "show me product",
                             "list product", "show the"]):
        return "navigation"
    if any(w in i for w in ["post", "create", "add", "write", "submit",
                             "draft", "reply", "subscribe", "rate",
                             "fill the", "notify", "re-post"]):
        return "create"
    if any(w in i for w in ["buy", "order"]):
        return "purchase"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="test_raw.json")
    ap.add_argument("--output", default="task_intents.json")
    args = ap.parse_args()

    data = json.load(open(args.input))
    
    # Filter to single-site tasks on target sites
    tasks = [t for t in data
             if len(t.get("sites", [])) == 1
             and t["sites"][0] in TARGET_SITES]

    # Build output structure
    output = {
        "summary": {
            "total_tasks": len(tasks),
            "by_site": {},
            "by_category": {},
        },
        "sites": {}
    }

    by_site = defaultdict(list)
    by_cat = defaultdict(int)

    for t in tasks:
        site = t["sites"][0]
        cat = categorise(t["intent"])
        by_site[site].append({
            "task_id": t["task_id"],
            "intent": t["intent"],
            "category": cat,
            "template": t.get("intent_template", ""),
            "eval_type": t.get("eval", {}).get("eval_types", []),
            "requires_state_change": t.get("require_reset", False),
        })
        by_cat[cat] += 1

    for site in ["shopping", "shopping_admin", "reddit"]:
        site_tasks = by_site[site]
        output["summary"]["by_site"][site] = len(site_tasks)
        
        # Group by category within site
        site_by_cat = defaultdict(list)
        for t in site_tasks:
            site_by_cat[t["category"]].append(t)
        
        output["sites"][site] = {
            "total": len(site_tasks),
            "categories": {}
        }
        for cat in sorted(site_by_cat):
            cat_tasks = site_by_cat[cat]
            # Deduplicate by template
            seen_templates = set()
            unique = []
            for t in cat_tasks:
                tmpl = t["template"] or t["intent"]
                if tmpl not in seen_templates:
                    seen_templates.add(tmpl)
                    unique.append(t)
            
            output["sites"][site]["categories"][cat] = {
                "count": len(cat_tasks),
                "distinct_templates": len(unique),
                "tasks": [{
                    "task_id": t["task_id"],
                    "intent": t["intent"],
                    "eval_type": t["eval_type"],
                } for t in unique]
            }

    output["summary"]["by_category"] = dict(sorted(by_cat.items(),
                                                     key=lambda x: -x[1]))

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)

    # Print summary
    print(f"Extracted {len(tasks)} tasks to {args.output}\n")
    print("By site:")
    for site, n in output["summary"]["by_site"].items():
        print(f"  {site:16s} {n:>3} tasks")
    print("\nBy category:")
    for cat, n in output["summary"]["by_category"].items():
        print(f"  {cat:25s} {n:>3} tasks")
    
    print("\n\nTask types the agent must handle:")
    print("=" * 70)
    for site in ["shopping", "shopping_admin", "reddit"]:
        print(f"\n--- {site.upper()} ---")
        for cat, info in output["sites"][site]["categories"].items():
            print(f"\n  {cat} ({info['count']} tasks, {info['distinct_templates']} types):")
            for t in info["tasks"]:
                print(f"    {t['task_id']:>4}  {t['intent'][:65]}")


if __name__ == "__main__":
    main()