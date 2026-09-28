from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import wa_env as W
import strategy_c
from spending_schemes import spending_scheme_config as C
from spending_schemes.allowance import allowance, prompt_char_limit, trim_prompt
from spending_schemes.agents import AGENTS, INSTRUCTION_TOKENS

TASKS_FILE = Path("../data/processed/task_list/read_only_tasks.jsonl")
RUN_SCHEMES = ["even", "front_loaded", "reactive"]
STRATEGIES = ["A", "B", "C"]

#run an episode for a given strategy and scheme
def run_episode(strategy: str, scheme: str, task_id: int, budget: int) -> dict:
    #budget is the total budget for the episode
    cap = budget
    site = W.site_of(task_id)
    #router is None for strategies A and B
    router = None
    p_cycle = None
    #if the strategy is C, use the router to choose the mode
    if strategy == "C":
        router, tasks = strategy_c._ready()
        task_row = {**tasks[task_id], "budget_level": cap}
        mode, p_cycle = router.choose_mode(task_row)
    else:
        mode = "execute" if strategy == "A" else "cycle"
    calls_per_step = 1 if mode == "execute" else 3

    print(f"\n{'=' * 60}\nTASK {task_id} ({site})  [{strategy} / {scheme} / {cap}]  mode={mode}"
          + (f"  p_cycle={p_cycle:.3f}" if p_cycle is not None else "") + f"\n{'=' * 60}")

    env = W.make_env(task_id)
    obs, _ = env.reset()
    goal = W.goal_of(obs)

    #initialize the token counts
    in_tok = out_tok = 0
    #steps is a list of dictionaries, one for each step
    steps: list[dict] = []
    #decisions is a list of dictionaries, one for each decision
    decisions: list[dict] = []
    #action_history is a list of strings, one for each action
    action_history: list[str] = []
    #url_history is a list of strings, one for each URL
    url_history: list[str] = [obs.get("url", "")]
    #url_action_counts is a Counter of the number of times each URL-action pair has occurred
    url_action_counts: Counter = Counter()
    #consecutive_errors is the number of consecutive errors
    consecutive_errors = 0
    #critic_revisions is the number of times the critic has revised the action
    critic_revisions = 0
    #last_error is a boolean, True if the last action was an error
    last_error = False
    #url_changed is a boolean, True if the URL has changed
    url_changed = False
    success = False
    reason = "max_steps"
    reward = 0.0
    stop_step = None
    trimmed_steps = 0
    t0 = time.time()

    #loop through the steps
    for i in range(W.MAX_STEPS):
        spent = in_tok + out_tok

        #if the strategy is C, use the router to check if the episode should stop
        if router is not None:
            state = {**task_row, "step_index": i, "budget_remaining_frac": 1 - spent / cap,
                     "last_error": int(last_error), "url_changed_last": int(url_changed),
                     "consecutive_errors": consecutive_errors, "mode": mode}
            p_stop = router.p_stop(state)
            stop = p_stop >= router.threshold
            decisions.append({"step": i, "p_stop": round(p_stop, 6), "stop": stop})
            if stop:
                stop_step, reason = i, "router_stop"
                break

        #how much this step may spend, at what effort
        step_tokens, effort, _ = allowance(scheme, i, cap, spent, last_error, url_changed)
        #limit is the maximum number of characters allowed in the prompt
        limit = prompt_char_limit(step_tokens, calls_per_step, INSTRUCTION_TOKENS[mode], effort)
        #full is the full prompt
        full = W.build_prompt(obs, action_history, url_history)
        #base is the base prompt
        base, trimmed = trim_prompt(full, limit)
        #trimmed_steps is the number of steps that were trimmed
        trimmed_steps += int(trimmed)
        #cur_url is the current URL
        cur_url = obs.get("url", "")
        #agents is the dictionary of agents for the given effort
        agents = AGENTS[effort]

        #est_step is the estimated number of tokens for the step
        est_step = calls_per_step * (len(base) // 4 + C.OUTPUT_MARGIN[effort]) + INSTRUCTION_TOKENS[mode]
        if spent + est_step > cap:
            reason = "budget_would_exceed"
            break
        #step_meta is a dictionary of metadata for the step
        step_meta = {"scheme": scheme, "allowance": step_tokens, "effort": effort,
                     "prompt_chars_full": len(full), "prompt_chars": len(base), "trimmed": trimmed}
        #print the step metadata
        print(f" step {i}  allowance {step_tokens:>6}  {effort:4s}  prompt {len(full)}->{len(base)}"
              f"{'  TRIMMED' if trimmed else ''}")
        #if the mode is execute, call the single agent
        if mode == "execute":
            result = W.call_agent(agents["single"], base)
            u = result.usage()
            in_tok += u.input_tokens
            out_tok += u.output_tokens
            final_action = result.output.action
            steps.append({"step": i, "agent_role": "single", "action": final_action, "url": cur_url,
                          "input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                          "cumulative_tokens": in_tok + out_tok,
                          "usage_details": dict(u.details) if getattr(u, "details", None) else None,
                          **step_meta})
            print(f"   action: {final_action}   (+{u.input_tokens}/{u.output_tokens}, cum {in_tok + out_tok})")
            if in_tok + out_tok >= cap:
                reason = "safety_token_cap"
                break
        else:
            def record(role, usage, action, extra=None):
                nonlocal in_tok, out_tok
                in_tok += usage.input_tokens
                out_tok += usage.output_tokens
                rec = {"step": i, "agent_role": role, "action": action, "url": cur_url,
                       "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
                       "cumulative_tokens": in_tok + out_tok,
                       "usage_details": dict(usage.details) if getattr(usage, "details", None) else None,
                       **step_meta}
                if extra:
                    rec.update(extra)
                steps.append(rec)

            #call the planner agent
            r_plan = W.call_agent(agents["planner"], base)
            #if the budget is exhausted, break
            if in_tok + out_tok >= cap:
                reason = "budget_exhausted_mid_step"
                break
            plan = r_plan.output.plan
            record("planner", r_plan.usage(), None, {"plan": plan})

            #call the executor agent
            low = AGENTS["low"]
            #if the budget is exhausted, break
            r_exec = W.call_agent(low["executor"], f"{base}\n\nPLANNER SUB-GOAL:\n{plan}")
            proposed = r_exec.output.action
            record("executor", r_exec.usage(), proposed, {"proposed_action": proposed})
            if in_tok + out_tok >= cap:
                reason = "budget_exhausted_mid_step"
                break

            #call the critic agent
            r_crit = W.call_agent(low["critic"], f"{base}\n\nPLANNER SUB-GOAL:\n{plan}\n\n"
                                                 f"EXECUTOR PROPOSED ACTION:\n{proposed}")
            #get the verdict from the critic agent
            verdict = r_crit.output
            #if the action is revised, increment the critic_revisions
            revised = (not verdict.approve) and bool(verdict.revised_action)
            #if the action is revised, use the revised action, otherwise use the proposed action
            final_action = verdict.revised_action if revised else proposed
            critic_revisions += int(revised)
            record("critic", r_crit.usage(), final_action,
                   {"approved": verdict.approve, "revised": revised, "proposed_action": proposed})
            print(f"   action: {final_action}   (cum {in_tok + out_tok})")
            if in_tok + out_tok >= cap:
                reason = "safety_token_cap"
                break

        #append the final action to the action history
        action_history.append(final_action)
        #pair is the current URL and the final action
        pair = (cur_url, final_action)
        #increment the url_action_counts
        url_action_counts[pair] += 1
        #step the environment
        obs, reward, terminated, truncated, _ = env.step(final_action)
        #new_url is the new URL
        new_url = obs.get("url", "")
        #append the new URL to the url history
        url_history.append(new_url)
        #err is the last action error
        err = obs.get("last_action_error")
        #increment the consecutive_errors if the last action was an error
        consecutive_errors = consecutive_errors + 1 if err else 0
        #set the last_error to True if the last action was an error
        last_error = bool(err)
        #set the url_changed to True if the new URL is different from the current URL
        url_changed = new_url != cur_url
        #update the steps
        for s in steps:
            if s["step"] == i:
                #set the action_error to the last action error
                s["action_error"] = str(err) if err else None
                #set the step_reward to the reward
                s["step_reward"] = reward
                #set the next_url to the new URL
                s["next_url"] = new_url
 
        #if the reward is 1.0, set the success to True and break
        if reward >= 1.0:
            success, reason = True, "success"
            break
        if terminated or truncated:
            reason = "env_terminated"
            break
        if consecutive_errors >= W.MAX_CONSECUTIVE_ERRORS:
            reason = "repeated_action_failure"
            break
        if url_action_counts[pair] >= W.MAX_SAME_ACTION_FROM_PAGE:
            reason = "navigation_cycle"
            break

    env.close()
    by_role = Counter()
    #count the tokens by role
    for s in steps:
        by_role[s["agent_role"]] += s["input_tokens"] + s["output_tokens"]

    rec = {
        "strategy": strategy, "scheme": scheme, "site": site, "task_id": task_id,
        "budget_level": cap, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "goal": goal, "success": success, "final_reward": reward,
        "termination_reason": reason, "steps": len({s["step"] for s in steps}),
        "agent_calls": len(steps), "mode_chosen": mode,
        "critic_revisions": critic_revisions if mode == "cycle" else None,
        "trimmed_steps": trimmed_steps, "tokens_by_role": dict(by_role),
        "input_tokens": in_tok, "output_tokens": out_tok, "total_tokens": in_tok + out_tok,
        "wall_clock_seconds": round(time.time() - t0, 1),
        "step_log": steps,
    }
    #if the strategy is C, add the router metadata
    if router is not None:
        rec["router"] = {"model_dir": router.model_dir, "stop_threshold": router.threshold,
                         "mode": mode, "p_cycle": round(p_cycle, 6),
                         "stopped_by_router": stop_step is not None, "stop_step": stop_step,
                         "decisions": decisions}
    return rec

#main function to run the schemes
def main() -> None:
    #parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../data/processed/schemes_output/schemes.jsonl")
    ap.add_argument("--schemes", nargs="+", default=RUN_SCHEMES)
    ap.add_argument("--strategies", nargs="+", default=STRATEGIES)
    ap.add_argument("--n", type=int, default=None, help="first N tasks only")
    ap.add_argument("--smoke", action="store_true", help="2 tasks, all cells")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    #load the tasks (one JSON object per line)
    budget = C.BUDGET
    task_ids = [json.loads(l)["task_id"] for l in TASKS_FILE.read_text().splitlines() if l.strip()]

    #configure strategy C
    if "C" in args.strategies:
        strategy_c.configure()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_path.exists():
        for line in out_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["strategy"], r["scheme"], r["task_id"]))

    #create the todo list of tasks to run
    todo = [(t, s, st) for t in task_ids for s in args.schemes for st in args.strategies
            if (st, s, t) not in done]
    print(f"[schemes] {len(task_ids)} tasks x {len(args.schemes)} schemes x {len(args.strategies)} "
          f"strategies = {len(task_ids) * len(args.schemes) * len(args.strategies)} cells; "
          f"{len(done)} done, {len(todo)} to run -> {out_path}")

    #run the episodes for the todo list of tasks
    t_start = time.time()
    with out_path.open("a") as fh:
        for k, (tid, scheme, strat) in enumerate(todo, 1):
            elapsed = (time.time() - t_start) / 60
            print(f"\n[schemes] {k}/{len(todo)}  task {tid}  {strat}/{scheme}  ({elapsed:.0f}m elapsed)")
            try:
                rec = run_episode(strat, scheme, tid, budget)
            except KeyboardInterrupt:
                print("[schemes] interrupted - completed episodes are saved")
                raise
            except Exception as e:
                import traceback
                traceback.print_exc()
                rec = {"strategy": strat, "scheme": scheme, "task_id": tid, "site": W.site_of(tid),
                       "budget_level": budget, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "error": f"{type(e).__name__}: {e}"}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    print(f"\n[schemes] finished in {(time.time() - t_start) / 60:.0f} minutes -> {out_path}")


if __name__ == "__main__":
    main()
