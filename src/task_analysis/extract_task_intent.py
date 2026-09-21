import argparse
import json
import re
from collections import defaultdict
 
TARGET_SITES = {"shopping", "shopping_admin", "reddit"}

#Pattern Groups

#if the task asks a question or requests information, it is read-only no matter what other nouns appear in the sentence.
READ_ONLY_PATTERNS = [
    r"show me", r"show all", r"show the", r"show my", r"show \d",
    r"tell me", r"tel+l me",
    r"what is", r"what are", r"what do", r"what'?s", r"what brands",
    r"which \w+",
    r"who has", r"who is",
    r"how many", r"how much", r"how long",
    r"provide me", r"give me",
    r"list out", r"list the", r"list all", r"list top", r"list \d",
    r"summari[sz]e",
    r"compare",
    r"count of", r"total number", r"grand total",
    r"among the", r"among \w+",
    r"is there any", r"are there any",
    r"preview",
    r"presents?\b",
    r"get the \w+ of",
]

# navigation and browsing, reaching a page or filtering a view without changing anything.
NAVIGATION_PATTERNS = [
    r"\bbrowse\b",
    r"\bsearch for\b",
    r"\blook up\b", r"\blookup\b",
    r"\bfind\b",
    r"\bnavigate\b",
    r"listings? by",
    r"products? under",
]

#bulk actions on multiple items.
BULK_ACTION_PATTERNS = [
    r"like all", r"dislike all",
    r"thumbs? (up|down)",
    r"upvote", r"downvote",
]


#deletions. Word boundary prevents "remover" matching.
DELETE_PATTERNS = [
    r"\bdelete\b", r"\bdeleting\b",
    r"\bremove\b", r"\bremoving\b",
]


#modifications to an existing value.
MODIFY_PATTERNS = [
    r"\breduce\b", r"\bincrease\b", r"\bdecrease\b",
    r"\bchange\b", r"\bupdate\b", r"\bmodify\b", r"\bedit\b",
    r"\bset the\b", r"\brename\b",
    r"\bmark all\b", r"\bmake all\b",
    r"\bdisable\b", r"\benable\b",
    r"\bcancel order\b", r"\bapprove\b",
    r"\bout of stock\b",
]


#creating new content or submitting forms.
CREATE_PATTERNS = [
    r"\bpost\b", r"\bposting\b", r"re-?post",
    r"\bcreate\b", r"\bdraft\b", r"\bwrite\b", r"\bsubmit\b",
    r"\breply\b", r"\bcomment\b",
    r"\badd\b", r"\bsubscribe\b",
    r"\brate\b", r"\brating\b",
    r"fill the", r"fill in",
    r"\bnotify\b",
    r"ask for",
]


#"Buy the best rating product" contains "rate", which CREATE_PATTERNS matches. "order" alone is too weak as a signal as it
#appears in many read-only queries ("the billing address for order 00178"),so only explicit buying verbs count.
PURCHASE_PATTERNS = [
    r"\bbuy\b", r"\bpurchase\b", r"\bcheckout\b", r"\bplace an order\b",
]

#helper function to check if the intent matches any of the patterns
def _matches(intent: str, patterns: list[str]) -> bool:
    return any(re.search(p, intent) for p in patterns)
 
#categorise the intent into one of the categories
def categorise(intent: str) -> str:
    i = (intent or "").lower()
 
    if _matches(i, READ_ONLY_PATTERNS):
        return "information_retrieval"
 
    if _matches(i, NAVIGATION_PATTERNS):
        return "navigation"
 
    if _matches(i, BULK_ACTION_PATTERNS):
        return "bulk_action"
 
    if _matches(i, DELETE_PATTERNS):
        return "delete"
 
    if _matches(i, MODIFY_PATTERNS):
        return "modify_value"
 
    if _matches(i, PURCHASE_PATTERNS):
        return "purchase"
 
    if _matches(i, CREATE_PATTERNS):
        return "create"
 
    return "other"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="test_raw.json")
    ap.add_argument("--output", default="task_intents.json")
    args = ap.parse_args()
 
 
    data = json.load(open(args.input))
 
    tasks = [t for t in data
             if len(t.get("sites", [])) == 1
             and t["sites"][0] in TARGET_SITES]
 
    output = {
        "summary": {
            "total_tasks": len(tasks),
            "by_site": {},
            "by_category": {},
        },
        "task_categories": {},
        "sites": {},
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
 
        site_by_cat = defaultdict(list)
        for t in site_tasks:
            site_by_cat[t["category"]].append(t)
 
        output["sites"][site] = {"total": len(site_tasks), "categories": {}}
        for cat in sorted(site_by_cat):
            cat_tasks = site_by_cat[cat]
            seen, unique = set(), []
            for t in cat_tasks:
                tmpl = t["template"] or t["intent"]
                if tmpl not in seen:
                    seen.add(tmpl)
                    unique.append(t)
 
            output["sites"][site]["categories"][cat] = {
                "count": len(cat_tasks),
                "distinct_templates": len(unique),
                "tasks": [{
                    "task_id": t["task_id"],
                    "intent": t["intent"],
                    "eval_type": t["eval_type"],
                } for t in unique],
            }
 
    output["summary"]["by_category"] = dict(
        sorted(by_cat.items(), key=lambda x: -x[1]))
 
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
 
    # ── Report ──────────────────────────────────────────────────────────
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
            print(f"\n  {cat} ({info['count']} tasks, "
                  f"{info['distinct_templates']} types):")
            for t in info["tasks"]:
                print(f"    {t['task_id']:>4}  {t['intent'][:65]}")
 
 
if __name__ == "__main__":
    main()