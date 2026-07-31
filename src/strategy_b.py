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

You are given the goal, the current page, and the actions taken so far.
Output the single next sub-goal in one short sentence.

- Be concrete about which element or region matters, e.g. "open the user's
  profile via the username link", not "find the user".
- Check the action history. Element ids change on every page load, so judge
  repetition by the URL an action led to, not by the id. If previous attempts
  have not changed the page, plan a DIFFERENT route.
- Returning to a hub page to take a different branch is legitimate. Repeatedly
  taking the same branch is not.
- Prefer navigating via links and scrolling over site search boxes: search
  endpoints are slow and often fail.
- If the information the task asks for is already visible on this page, say
  "the answer is on this page: <value>" - the Executor will then submit it.
- If the task only requires REACHING a page and you are already on it, say
  "the target page has been reached; stay here".
- If the task cannot be completed on this site at all, say so explicitly.
- If the task cannot be completed on this site at all (you have checked and
  confirmed, not just guessed), say "this task is impossible on this site"
  and the Executor will submit N/A.
- If the action history shows select_option was tried on an element without
  changing the page, the control is a custom widget, not a standard dropdown.
  Plan to click it open instead, then look for the option in the next
  observation or type into a searchbox.
- When filling a form with multiple fields, complete one field at a time and
  verify it took effect before moving to the next. Do not skip ahead to other
  fields if a required field (like a forum/category selector) still shows its
  default value.
"""


EXECUTOR_INSTRUCTIONS = f"""\
You are the EXECUTOR in a three-agent web-navigation pipeline. A Planner has
given you one sub-goal for this step. Translate it into exactly one action.

{_ACTION_VOCAB}

- Only use element ids that appear in the current AXTree.
- Do exactly what the sub-goal asks. Do not pursue a different route.
- For standard dropdowns, use select_option('id', 'value'). If the sub-goal
  says to click a widget open or type into a searchbox, do that instead.
- If the sub-goal says the answer is on this page, submit it with
  send_msg_to_user. The answer is graded by EXACT MATCH: send ONLY the value,
  with no explanation, preamble or surrounding sentence.
      Correct:   send_msg_to_user('Sprite Stasis Ball 65 cm')
      Wrong:     send_msg_to_user('The top seller is Sprite Stasis Ball 65 cm')
  Numbers as digits only. Multiple items comma-separated. If nothing satisfies
  the criteria, or the task is impossible on this site, send 'N/A'.
- If the sub-goal says the target page has been reached, use noop().
- If the previous action reported a page error (500, 502, 504), use go_back().
"""


CRITIC_INSTRUCTIONS = f"""\
You are the CRITIC in a three-agent web-navigation pipeline. You are shown the
goal, the current page, the Planner's sub-goal and the Executor's proposed
action. Decide whether that action should be executed.

{_ACTION_VOCAB}

Approve unless there is a concrete problem:
- the element id does not appear in the current AXTree
- the action does not serve the stated sub-goal
- the action repeats something that has already failed on this page
- select_option or fill is proposed on an element where the same action type
  on the same element already appears in the action history at the same URL
  without the page changing - it failed before and will fail again. Suggest
  a different element or a different action type
- an answer is malformed: wrapped in a sentence, explained, quoted, or not an
  exact match to what the task asks for
- an answer is being submitted before the information has actually been
  verified on the page (submitting ends the episode and cannot be undone)
- the task only required reaching a page, that page has been reached, and the
  action would navigate away from it
- the page shows a server error (500, 502, 504) but the proposed action is not
  go_back()

If you reject, supply revised_action with a valid replacement drawn from the
current AXTree. Do not reject merely because you would have chosen a different
route - only when the proposed action is wrong.
"""


planner = W.make_agent(PLANNER_INSTRUCTIONS, PlannerPlan, label="planner")
executor = W.make_agent(EXECUTOR_INSTRUCTIONS, ExecutorAction, label="executor")
critic = W.make_agent(CRITIC_INSTRUCTIONS, CriticVerdict, label="critic")


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
        est_step = (len(base) // 4) * 3 + 1500
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