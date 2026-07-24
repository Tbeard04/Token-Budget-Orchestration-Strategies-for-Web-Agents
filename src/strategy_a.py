"""
strategy_a.py - Strategy A: single-agent baseline.

One LLM call per step. Establishes the cost-performance floor against which
Strategies B and C are measured.

This is a token-minimised adaptation of WebArena's chain-of-thought baseline
agent: few-shot exemplars are omitted and reasoning length is constrained,
because a two-shot CoT prompt costs 1-2k tokens per call and is therefore
unaffordable under the lower budget conditions this study examines.

Run:
    python strategy_a.py 276              # one task at the default budget
    python strategy_a.py --budget 8000 47 276
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

import wa_env as W


DEFAULT_BUDGET = 16_000


# ----------------------------------------------------------------------------
# Structured output: the model is constrained to emit a valid shape, so no
# regex extraction of the action is needed.
# ----------------------------------------------------------------------------
class AgentAction(BaseModel):
    reasoning: str = Field(description="One short sentence justifying the action")
    action: str = Field(description="A single BrowserGym action string")


INSTRUCTIONS = """\
You are a web-navigation agent operating a website through BrowserGym.
You are given a task goal and the page's accessibility tree (AXTree). Each
element has a bracketed id such as [a31]. Return EXACTLY ONE action string,
using the id WITHOUT the brackets:

  click('a31')                  - click an element
  fill('b12', 'some text')      - type into a field
  select_option('c4', 'Blue')   - choose a dropdown option
  press('a31', 'Enter')         - press a key
  scroll(0, 400)                - scroll the page
  goto('http://...')            - navigate to a URL
  go_back()                     - browser back
  send_msg_to_user('answer')    - ANSWER an information-seeking task
  noop()                        - do nothing

Rules:
- Only use ids that appear in the current AXTree.
- If the task asks a question, it is complete only once you call
  send_msg_to_user(...) with the answer.
- Keep reasoning to one sentence. Output one action only.
- Review ACTIONS YOU HAVE ALREADY TAKEN. Each line shows an action and the URL
  it led to. Element ids change on every page load, so the SAME destination can
  have a DIFFERENT id - judge repetition by the URL, not the id. If a URL
  appears repeatedly WITHOUT you reaching any new page in between, you are
  going in circles: try a different route or a different part of the page.
  Returning to a hub page to take a different branch is fine.
- Prefer navigating via links and scrolling over using site search boxes.
  Search endpoints are slow and often fail. Only use search if no navigational
  path is visible.
- If the page shows a server error (500, 502, 504) or fails to load, use
  go_back() and try a different route. Do NOT answer N/A because of a page
  error - N/A means the information genuinely does not exist.
- If the task is impossible to complete on this site (the data does not exist,
  or the site does not support the requested operation), send exactly:
  send_msg_to_user('N/A')
- For dropdowns, comboboxes and select elements, use
  select_option('378', 'games') directly with the option you want. Do NOT
  click a dropdown to "open" it first - clicking does nothing useful and
  wastes a step.
- Some tasks only require you to REACH a page ("browse X", "search for Y",
  "go to Z"). These need no answer: once the correct page is loaded the task
  is complete. Do not click into individual items or navigate away - use
  noop() to stay on the page.

Answering with send_msg_to_user - the answer is graded by EXACT MATCH:
- Send ONLY the answer itself. No explanation, no preamble, no quotes,
  no sentence wrapping it.
  Correct:   send_msg_to_user('Sprite Stasis Ball 65 cm')
  Wrong:     send_msg_to_user('The top seller is Sprite Stasis Ball 65 cm')
- For a list of items, comma-separate them: send_msg_to_user('Alice, Bob')
- If nothing on the page satisfies the criteria, send exactly:
  send_msg_to_user('N/A')
- Numbers: send digits only, e.g. send_msg_to_user('0')
- Do NOT answer until you have navigated to and verified the specific
  information the task asks for. Answering early ends the episode and
  cannot be undone.
"""

agent = W.make_agent(INSTRUCTIONS, AgentAction, label="strategy_a")


# ----------------------------------------------------------------------------
def run_episode(task_id: int, budget: int | None = None) -> dict:
    """Run one Strategy A episode under a hard token budget."""
    cap = budget if budget is not None else DEFAULT_BUDGET
    site = W.site_of(task_id)
    print(f"\n{'=' * 60}\nTASK {task_id}  (site: {site})  [A, budget {cap}]\n{'=' * 60}")

    env = W.make_env(task_id)
    obs, _ = env.reset()

    goal = W.goal_of(obs)
    print(f"Goal: {goal}")
    print(f"Start URL: {obs.get('url', 'unknown')}\n")

    in_tok = out_tok = 0
    steps: list[dict] = []
    action_history: list[str] = []
    url_history: list[str] = [obs.get("url", "")]
    url_action_counts: Counter = Counter()
    consecutive_errors = 0
    success = False
    reason = "max_steps"
    reward = 0.0
    t0 = time.time()

    for i in range(W.MAX_STEPS):
        prompt = W.build_prompt(obs, action_history, url_history)

        # Check BEFORE paying: an episode must never exceed its stated budget.
        spent = in_tok + out_tok
        est_in = len(prompt) // 4
        if spent + est_in > cap:
            reason = "budget_would_exceed"
            print(f"\n step {i}: SKIPPED - est. {est_in} input tokens would exceed "
                  f"cap (used {spent}/{cap})")
            break

        result = W.call_agent(agent, prompt)
        u = result.usage()

        in_tok += u.input_tokens
        out_tok += u.output_tokens
        total = in_tok + out_tok
        decided = result.output
        action_history.append(decided.action)

        # Count this action taken FROM this page (obs is still pre-step here)
        pair = (obs.get("url", ""), decided.action)
        url_action_counts[pair] += 1
        pair_n = url_action_counts[pair]

        print(f"\n step {i}")
        print(f"   reason : {decided.reasoning}")
        print(f"   action : {decided.action}")
        print(f"   url    : {obs.get('url', '')[:90]}")
        print(f"   tokens : +{u.input_tokens} in / +{u.output_tokens} out"
              f"   (cumulative {total})")
        if getattr(u, "details", None):
            print(f"   detail : {u.details}")

        steps.append({
            "step": i,
            "agent_role": "single",
            "action": decided.action,
            "url": obs.get("url", ""),
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cumulative_tokens": total,
            "usage_details": dict(u.details) if getattr(u, "details", None) else None,
        })

        if total >= cap:
            reason = "safety_token_cap"
            break

        obs, reward, terminated, truncated, _ = env.step(decided.action)
        url_history.append(obs.get("url", ""))
        err = obs.get("last_action_error")
        consecutive_errors = consecutive_errors + 1 if err else 0

        print(f"   result : reward={reward}"
              f"{'  ERROR: ' + str(err) if err else '  (action accepted)'}")

        if reward >= 1.0:
            success, reason = True, "success"
            break
        if terminated or truncated:
            reason = "env_terminated"
            break

        # Shared environment constraints (identical for A, B and C).
        # NB: URL change is deliberately NOT the progress signal - AJAX
        # workflows (admin grids, filter panels, forms) legitimately operate
        # on a single URL for many steps.
        if consecutive_errors >= W.MAX_CONSECUTIVE_ERRORS:
            reason = "repeated_action_failure"
            print(f"   STALL  : {consecutive_errors} consecutive failed actions")
            break
        if pair_n >= W.MAX_SAME_ACTION_FROM_PAGE:
            reason = "navigation_cycle"
            print(f"   CYCLE  : same action from same page {pair_n}x")
            break

    env.close()

    record = {
        "strategy": "A",
        "site": site,
        "task_id": task_id,
        "budget_level": cap,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "goal": goal,
        "success": success,
        "final_reward": reward,
        "termination_reason": reason,
        "steps": len(steps),
        "agent_calls": len(steps),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "wall_clock_seconds": round(time.time() - t0, 1),
        "step_log": steps,
    }

    print(f"\n --- A / {site} / task {task_id} ---")
    for k in ("success", "termination_reason", "steps",
              "input_tokens", "output_tokens", "total_tokens",
              "wall_clock_seconds"):
        print(f"   {k:20s}: {record[k]}")
    return record


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", type=int,
                    help="task ids; default is one per site")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--out", default="../data/raw/strategy_a_results.jsonl")
    args = ap.parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.tasks:
        task_ids = args.tasks
    else:
        pools = W.single_site_tasks()
        task_ids = [ids[0] for ids in pools.values() if ids]
        print(f"No tasks given; using one per site: {task_ids}")

    results = []
    with open(args.out, "w") as fh:
        for tid in task_ids:
            try:
                rec = run_episode(tid, args.budget)
            except Exception as e:
                import traceback
                traceback.print_exc()
                rec = {"strategy": "A", "task_id": tid, "site": W.site_of(tid),
                       "error": f"{type(e).__name__}: {e}"}
            results.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    print(f"\n{'=' * 78}\nSUMMARY\n{'=' * 78}")
    print(f"{'site':16s}{'task':>6s}{'ok':>7s}{'steps':>7s}{'tokens':>9s}"
          f"{'secs':>8s}  reason")
    for r in results:
        if "error" in r:
            print(f"{r['site']:16s}{r['task_id']:>6}{'ERR':>7s}"
                  f"{'-':>7s}{'-':>9s}{'-':>8s}  {r['error'][:34]}")
        else:
            print(f"{r['site']:16s}{r['task_id']:>6}{str(r['success']):>7s}"
                  f"{r['steps']:>7}{r['total_tokens']:>9}"
                  f"{r['wall_clock_seconds']:>8}  {r['termination_reason']}")

    print(f"\nWrote {len(results)} episodes to {args.out}")


if __name__ == "__main__":
    main()
