"""
strategy_b.py - Strategy B: fixed Planner -> Executor -> Critic pipeline.

Three LLM calls per logical step, on every step, regardless of task complexity
or remaining budget. That budget-blindness is deliberate: it is the property
Strategy C's learned router is designed to remove.

PROMPTING
Each role has its own instructions, following standard practice for pipeline
architectures. Only the ACTION VOCABULARY is shared with Strategy A, because
that is the environment's interface rather than a prompt choice and must be
identical everywhere. The A-vs-B comparison therefore reflects both the
addition of agents and their specialisation; together these constitute the
"pipeline" condition. State this in the methodology.

ONE LOGICAL STEP
    1. Planner  sees goal + observation + history      -> a sub-goal in words
    2. Executor sees the above + the sub-goal          -> a BrowserGym action
    3. Critic   sees the above + the proposed action   -> approve or revise
    4. Executed action = the Critic's revision if it rejected, else the
       Executor's proposal

The budget is checked before the step AND between roles, so a step can abort
part-way through. Episodes ending mid-step - having paid for a Planner and an
Executor without ever acting - are budget-blindness made visible, and are
recorded as 'budget_exhausted_mid_step'.

LOGGING
THREE step records per logical step, one per role, each with its own token
counts. This per-role cost record is the training signal for Strategy C:
without it the router cannot learn what invoking each agent costs, or when
invoking it changes the outcome.

Run:
    python strategy_b.py 623                     # one task, default budget
    python strategy_b.py --budget 16000 47 276
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from pydantic import BaseModel, Field

import wa_env as W
import strategy_a


DEFAULT_BUDGET = 16_000


# ----------------------------------------------------------------------------
# Structured outputs - one per role.
# Each is a contract: the Planner cannot emit an action, the Executor cannot
# emit a plan, and the Critic must commit to approve/reject explicitly.
# ----------------------------------------------------------------------------
class PlannerPlan(BaseModel):
    reasoning: str = Field(description="One sentence of analysis")
    plan: str = Field(
        description="The next sub-goal in plain English, NOT an action string"
    )


class ExecutorAction(BaseModel):
    reasoning: str = Field(description="One sentence justifying the action")
    action: str = Field(description="A single BrowserGym action string")


class CriticVerdict(BaseModel):
    approve: bool = Field(
        description="True if the proposed action should be executed unchanged"
    )
    reasoning: str = Field(description="One sentence explaining the verdict")
    revised_action: str | None = Field(
        default=None,
        description="A corrected BrowserGym action string, only when approve is False",
    )


# ----------------------------------------------------------------------------
# Role instructions
#
# _ACTION_VOCAB is the part of Strategy A's instructions before "Rules:" - the
# environment interface. Everything after it here is role-specific: the Planner
# never emits an action so needs no answer-formatting rules; the Executor never
# chooses strategy so needs no exploration heuristics.
# ----------------------------------------------------------------------------
_ACTION_VOCAB = strategy_a.INSTRUCTIONS.split("Rules:")[0]


PLANNER_INSTRUCTIONS = f"""\
You are the PLANNER in a three-agent web-navigation pipeline. You decide what
should happen next. You do NOT produce actions and you do NOT answer the task.
 
{_ACTION_VOCAB}
 
Given the goal, current page, and action history, output the next sub-goal in
one short sentence.
 
NAVIGATION
- Name the specific element or region, e.g. "open the user's profile via the
  username link", not "find the user".
- Element ids change on every page load. Judge repetition by the URL an action
  led to, not the id. If previous attempts have not changed the page, plan a
  DIFFERENT route.
- Returning to a hub page to take a different branch is legitimate; repeating
  the same branch is not.
- Prefer links and scrolling over search boxes; search endpoints often fail.
- If select_option was tried on an element without the page changing, it is a
  custom widget. Plan to click it open, then look for the option or a
  searchbox in the next observation.
 
INFORMATION TASKS
- If the answer is visible on this page, say "the answer is on this page:
  <value>" and the Executor will submit it.
- For counting, summing, or comparing across items, ensure ALL relevant items
  are visible first - scroll or paginate before answering.
- On admin pages, navigate to the correct section and apply filters BEFORE
  reading. Dashboard summaries may not match the required period or status.
 
NAVIGATION-ONLY TASKS
- If the task only requires reaching a page ("browse X", "search for Y") and
  you are on it, say "the target page has been reached; stay here". Do not
  click into individual items.
- Apply any required sort or filter BEFORE declaring the target reached.
 
STATE-CHANGE TASKS
- Calculate the target value ONCE from the ORIGINAL value on the page. After
  the action history shows the field was filled, plan to SAVE. Do NOT re-read
  and re-apply the modification - that compounds the change.
- Complete one field at a time; do not move on while a required field still
  shows its default.
- Once all fields are filled, the next sub-goal is "click the Save button".
  Identify it by label (Save, Submit, Post, Create), not position - a
  "Submit" link in the site header is navigation, not the form's submit.
 
CREATE TASKS
- Identify all required fields first, then fill them in order, then submit.
- For forum posts, select the forum FIRST and confirm it before title/body.
- For rating or review tasks, navigate to the product before looking for the
  review form.
 
BULK ACTIONS
- For "like all" / "dislike all" tasks, reach the page listing all items, then
  act on each sequentially, scrolling to reveal more.
 
DELETE TASKS
- Some admin views require checkbox selection then a bulk action - plan for
  that pattern rather than looking for a per-row delete button.
 
IMPOSSIBLE TASKS
- If you have checked and confirmed the task cannot be done on this site, say
  "this task is impossible on this site" and the Executor will submit N/A.
 
ERROR RECOVERY
- On a server error (500, 502, 504) or a page that fails to load, plan to go
  back and try a different route.
"""


EXECUTOR_INSTRUCTIONS = f"""\
You are the EXECUTOR in a three-agent web-navigation pipeline. A Planner has
given you one sub-goal. Translate it into exactly one action.
 
{_ACTION_VOCAB}
 
- Only use element ids that appear in the current AXTree.
- Do exactly what the sub-goal asks. Do not pursue a different route.
- Use the exact value the sub-goal specifies. Do not recalculate or rephrase.
 
DROPDOWNS AND WIDGETS
- For standard dropdowns use select_option. If the sub-goal says to click a
  widget open or type into a searchbox, do that instead.
 
FORM FILLING
- Put multi-line text (descriptions, post bodies) in a single fill() call.
- Identify save/submit buttons by their label (Save, Submit, Post, Create).
  A "Submit" link in the site header navigates away - it is not the form's
  submit button.
 
ANSWERING
- Graded by EXACT MATCH. Send ONLY the value, no explanation or sentence.
      Correct:   send_msg_to_user('Sprite Stasis Ball 65 cm')
      Wrong:     send_msg_to_user('The top seller is Sprite Stasis Ball 65 cm')
- Numbers as digits only. Lists comma-separated. Include currency symbols if
  shown on the page. If nothing matches, or the task is impossible, send 'N/A'.
 
NAVIGATION
- If the sub-goal says the page has been reached, use noop().
- On a page error (500, 502, 504), use go_back().
"""


CRITIC_INSTRUCTIONS = f"""\
You are the CRITIC in a three-agent web-navigation pipeline. You are shown the
goal, current page, the Planner's sub-goal and the Executor's proposed action.
Decide whether it should be executed.
 
{_ACTION_VOCAB}
 
FIRST: confirm the element id in the proposed action actually appears in the
current AXTree. Do not claim an element is present without checking -
approving actions on non-existent elements is the most common failure.
 
Approve unless there is a concrete problem:
 
ELEMENT PROBLEMS
- The element id is not in the current AXTree.
- The action uses syntax outside the vocabulary above.
- select_option or fill is proposed on an element where the same action type
  already appears in the action history at this URL without the page changing.
 
STATE-CHANGE PROBLEMS
- The same field is being filled with a DIFFERENT value than a previous fill
  on that element - the agent is re-applying a modification that was already
  made. Suggest saving instead.
- A form is being submitted with a required field still empty or at default.
- The action clicks a header navigation link instead of the form's submit.
 
ANSWER PROBLEMS
- The answer is malformed: wrapped in a sentence, explained, or not an exact
  match to what the task asks for.
- The answer is submitted before the information was verified on the page, or
  counted from a partial list.
 
NAVIGATION PROBLEMS
- The action does not serve the stated sub-goal.
- The target page was reached but the action navigates away from it.
- The page shows a server error but the action is not go_back().
 
If you reject, supply revised_action drawn from the current AXTree. Do not
reject because you would have chosen a different route - only when the action
is wrong.
"""


planner = W.make_agent(PLANNER_INSTRUCTIONS, PlannerPlan, label="planner")
executor = W.make_agent(EXECUTOR_INSTRUCTIONS, ExecutorAction, label="executor")
critic = W.make_agent(CRITIC_INSTRUCTIONS, CriticVerdict, label="critic")

# QUICK CHECK ON INSTANCE TO SEE IF ACTION VOCABULARY IS VISIBLE TO AGENTS
for _label, _instr in [("planner", PLANNER_INSTRUCTIONS),
                       ("executor", EXECUTOR_INSTRUCTIONS),
                       ("critic", CRITIC_INSTRUCTIONS)]:
    if "click('a31')" not in _instr:
        raise RuntimeError(
            f"{_label} instructions do not contain the action vocabulary. "
            f"Check for doubled braces: {{_ACTION_VOCAB}} escapes the "
            f"f-string interpolation and must be {{_ACTION_VOCAB}} singly."
        )

_INSTRUCTION_TOKENS = (
    len(PLANNER_INSTRUCTIONS)
    + len(EXECUTOR_INSTRUCTIONS)
    + len(CRITIC_INSTRUCTIONS)
) // 4
_OUTPUT_MARGIN = 900

# ----------------------------------------------------------------------------
def run_episode(task_id: int, budget: int | None = None) -> dict:
    """Run one Strategy B episode under a hard token budget.

    Signature matches strategy_a.run_episode so run_batch can swap them.
    """
    cap = budget if budget is not None else DEFAULT_BUDGET
    site = W.site_of(task_id)
    print(f"\n{'=' * 60}\nTASK {task_id}  (site: {site})  [B, budget {cap}]\n{'=' * 60}")

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
    critic_revisions = 0
    success = False
    reason = "max_steps"
    reward = 0.0
    t0 = time.time()

    def record(step_i: int, role: str, usage, action: str | None, url: str,
               extra: dict | None = None) -> None:
        """Log one agent call. Three of these per logical step."""
        nonlocal in_tok, out_tok
        in_tok += usage.input_tokens
        out_tok += usage.output_tokens
        rec = {
            "step": step_i,
            "agent_role": role,
            "action": action,
            "url": url,
            "prompt_chars": len(base),
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cumulative_tokens": in_tok + out_tok,
            "usage_details": dict(usage.details) if getattr(usage, "details", None) else None,
        }
        if extra:
            rec.update(extra)
        steps.append(rec)
        print(f"   {role:8s}: +{usage.input_tokens} in / +{usage.output_tokens} out"
              f"   (cumulative {in_tok + out_tok})")

    for i in range(W.MAX_STEPS):
        base = W.build_prompt(obs, action_history, url_history)
        cur_url = obs.get("url", "")

        # Pre-step check. Committing to a step means committing to three calls,
        # so estimate three prompts plus a margin for the appended plan/action
        # text and the three outputs.
        spent = in_tok + out_tok
        est_step = (len(base) // 4) * 3 + _INSTRUCTION_TOKENS + _OUTPUT_MARGIN
        if spent + est_step > cap:
            reason = "budget_would_exceed"
            print(f"\n step {i}: SKIPPED - est. {est_step} tokens for a 3-agent "
                  f"step would exceed cap (used {spent}/{cap})")
            break

        print(f"\n step {i}")
        print(f"   url     : {cur_url[:90]}")

        # --- 1. Planner: what should happen next -----------------------------
        r_plan = W.call_agent(planner, base)
        if in_tok + out_tok >= cap:
            reason = "budget_exhausted_mid_step"
            print("   STOP    : budget exhausted after the Planner")
            break
        plan = r_plan.output.plan
        record(i, "planner", r_plan.usage(), None, cur_url,{"plan": plan})
        print(f"   plan    : {plan}")

        # --- 2. Executor: turn the sub-goal into one action -------------------
        exec_prompt = f"{base}\n\nPLANNER SUB-GOAL:\n{plan}"
        r_exec = W.call_agent(executor, exec_prompt)
        proposed = r_exec.output.action
        record(i, "executor", r_exec.usage(), proposed, cur_url,{"proposed_action": proposed})
        if in_tok + out_tok >= cap:
            reason = "budget_exhausted_mid_step"
            print("   STOP    : budget exhausted after the Executor")
            break
        print(f"   proposed: {proposed}")

        # --- 3. Critic: approve or revise ------------------------------------
        crit_prompt = (f"{base}\n\nPLANNER SUB-GOAL:\n{plan}\n\n"
                       f"EXECUTOR PROPOSED ACTION:\n{proposed}")
        r_crit = W.call_agent(critic, crit_prompt)
        verdict = r_crit.output
        revised = (not verdict.approve) and bool(verdict.revised_action)
        final_action = verdict.revised_action if revised else proposed
        if revised:
            critic_revisions += 1
        record(i, "critic", r_crit.usage(), final_action, cur_url,
               {"approved": verdict.approve,
                "revised": revised,
                "proposed_action": proposed,
                "critic_reasoning": verdict.reasoning})
        print(f"   critic  : {'REVISED' if revised else 'approved'} -> {final_action}")
        if revised:
            print(f"   why     : {verdict.reasoning}")

        if in_tok + out_tok >= cap:
            reason = "safety_token_cap"
            break

        # --- 4. Execute -------------------------------------------------------
        action_history.append(final_action)
        pair = (cur_url, final_action)
        url_action_counts[pair] += 1
        pair_n = url_action_counts[pair]

        obs, reward, terminated, truncated, _ = env.step(final_action)
        url_history.append(obs.get("url", ""))
        err = obs.get("last_action_error")
        consecutive_errors = consecutive_errors + 1 if err else 0

        #Attach the outcome to all three role records for this step.
        #Required by Strategy C: the router must know "the previous action failed" to decide whether to invoke the Critic or Stop.
        for s in steps:
            if s["step"] == i:
                s["action_error"] = str(err) if err else None
                s["step_reward"] = reward
                s["next_url"] = obs.get("url", "")

        print(f"   result  : reward={reward}"
              f"{'  ERROR: ' + str(err)[:70] if err else '  (action accepted)'}")

        if reward >= 1.0:
            success, reason = True, "success"
            break
        if terminated or truncated:
            reason = "env_terminated"
            break

        # Shared environment constraints - identical thresholds to A and C.
        # NB: URL change is deliberately NOT the progress signal - AJAX
        # workflows (admin grids, filter panels, forms) legitimately operate
        # on a single URL for many steps.
        if consecutive_errors >= W.MAX_CONSECUTIVE_ERRORS:
            reason = "repeated_action_failure"
            print(f"   STALL   : {consecutive_errors} consecutive failed actions")
            break
        if pair_n >= W.MAX_SAME_ACTION_FROM_PAGE:
            reason = "navigation_cycle"
            print(f"   CYCLE   : same action from same page {pair_n}x")
            break

    env.close()

    logical_steps = len({s["step"] for s in steps})
    by_role = {"planner": 0, "executor": 0, "critic": 0}
    for s in steps:
        by_role[s["agent_role"]] += s["input_tokens"] + s["output_tokens"]

    record_out = {
        "strategy": "B",
        "site": site,
        "task_id": task_id,
        "budget_level": cap,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "goal": goal,
        "success": success,
        "final_reward": reward,
        "termination_reason": reason,
        "steps": logical_steps,
        "agent_calls": len(steps),
        "critic_revisions": critic_revisions,
        "tokens_by_role": by_role,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "wall_clock_seconds": round(time.time() - t0, 1),
        "step_log": steps,
    }

    print(f"\n --- B / {site} / task {task_id} ---")
    for k in ("success", "termination_reason", "steps", "agent_calls",
              "critic_revisions", "input_tokens", "output_tokens",
              "total_tokens", "wall_clock_seconds"):
        print(f"   {k:22s}: {record_out[k]}")
    print(f"   {'tokens_by_role':22s}: {by_role}")
    return record_out


# ----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="+", type=int, help="task ids")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--out", default="../data/raw/strategy_b_results.jsonl")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = []
    with out_path.open("w") as fh:
        for tid in args.tasks:
            try:
                rec = run_episode(tid, args.budget)
            except Exception as e:
                import traceback
                traceback.print_exc()
                rec = {"strategy": "B", "task_id": tid, "site": W.site_of(tid),
                       "error": f"{type(e).__name__}: {e}"}
            results.append(rec)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

    print(f"\n{'=' * 88}\nSUMMARY\n{'=' * 88}")
    print(f"{'site':16s}{'task':>6s}{'ok':>7s}{'steps':>7s}{'calls':>7s}"
          f"{'revis':>7s}{'tokens':>9s}{'secs':>8s}  reason")
    for r in results:
        if "error" in r:
            print(f"{r['site']:16s}{r['task_id']:>6}{'ERR':>7s}"
                  f"{'-':>7s}{'-':>7s}{'-':>7s}{'-':>9s}{'-':>8s}  {r['error'][:30]}")
        else:
            print(f"{r['site']:16s}{r['task_id']:>6}{str(r['success']):>7s}"
                  f"{r['steps']:>7}{r['agent_calls']:>7}{r['critic_revisions']:>7}"
                  f"{r['total_tokens']:>9}{r['wall_clock_seconds']:>8}"
                  f"  {r['termination_reason']}")

    print(f"\nWrote {len(results)} episodes to {out_path}")


if __name__ == "__main__":
    main()