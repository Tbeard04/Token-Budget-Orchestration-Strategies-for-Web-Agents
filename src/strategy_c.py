"""
strategy_c.py - Strategy C: router-directed dynamic strategy.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

import wa_env as W
import strategy_a
import strategy_b
from strategy_c.train.train_router import MLP, task_features, stop_features


DEFAULT_BUDGET = 16_000
#Path to the router-directed dynamic strategy model
ROUTER_DIR = Path("../data/processed/strategy_c_models/model_2")
#Path to the task metadata
TASK_METADATA = Path("../data/processed/final_annotation_difficulty_tiers/task_metadata.jsonl")
#Path to the task risk levels
TASK_RISK = Path("../data/processed/task_list/task_risk_levels.jsonl")
#Ordinal mapping for task difficulty tiers
TIER_ORD = {"Easy": 0, "Medium": 1, "Hard": 2}

#What a router-initiated stop submits. "none": end the episode with no answer
STOP_ANSWER = "none"


#router class for the router-directed dynamic strategy
class Router:
    def __init__(self, model_dir: Path):
        #load the router model
        ck_path = model_dir / "router.pt"
        #load the router model
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        #get the hidden layer size
        hidden = ck["config"]["hidden"]
        #create the mode network
        self.mode_net = MLP(len(ck["mode_features"]), hidden)
        #load the mode network state
        self.mode_net.load_state_dict(ck["mode_state"])
        #create the stop network
        self.stop_net = MLP(len(ck["stop_features"]), hidden)
        #load the stop network state
        self.stop_net.load_state_dict(ck["stop_state"])
        #set the mode network to evaluation mode
        self.mode_net.eval()
        #set the stop network to evaluation mode
        self.stop_net.eval()
        #get the mode mean
        self.mode_mean = np.array(ck["mode_norm"]["mean"], np.float32)
        #get the mode standard deviation
        self.mode_std = np.array(ck["mode_norm"]["std"], np.float32)
        #get the stop mean
        self.stop_mean = np.array(ck["stop_norm"]["mean"], np.float32)
        #get the stop standard deviation
        self.stop_std = np.array(ck["stop_norm"]["std"], np.float32)
        #get the stop threshold
        self.threshold = ck["stop_threshold"]
        #get the alpha
        self.alpha = ck["config"]["alpha"]
        #set the model directory
        self.model_dir = str(model_dir)
        #print the router loaded from the model directory
        print(f"[strategy_c] router loaded from {model_dir}  alpha={self.alpha}  "
              f"stop_threshold={self.threshold:.6f}")
 
    #helper function to compute the probability of a network output
    def _p1(self, net: MLP, x: list[float], mean, std) -> float:
        #normalize the input
        z = (np.array(x, np.float32) - mean) / std
        #compute the probability of the network output
        with torch.no_grad():
            return float(torch.softmax(net(torch.tensor(z)[None]), -1)[0, 1])
 
    #choose the mode based on the task features
    def choose_mode(self, task_row: dict) -> tuple[str, float]:
        p_cycle = self._p1(self.mode_net, task_features(task_row), self.mode_mean, self.mode_std)
        return ("cycle" if p_cycle >= 0.5 else "execute"), p_cycle
 
    #compute the probability of a stop based on the state features
    def p_stop(self, state_row: dict) -> float:
        return self._p1(self.stop_net, stop_features(state_row), self.stop_mean, self.stop_std)
 
 #global router and tasks variables
_router: Router | None = None
#global tasks variable
_tasks: dict[int, dict] | None = None

#configure the router and tasks
def configure(router_dir: str | Path | None = None, stop_answer: str | None = None) -> None:
    global _router, _tasks, STOP_ANSWER
    #if the stop answer is not None, check if it is valid
    if stop_answer is not None:
        if stop_answer not in ("none", "na"):
            raise ValueError("stop_answer must be 'none' or 'na'")
        STOP_ANSWER = stop_answer
    #load the router
    _router = Router(Path(router_dir) if router_dir else ROUTER_DIR)
    #load the tasks
    _tasks = load_task_rows()

#load the task rows
def load_task_rows() -> dict[int, dict]:
    #load the task metadata
    meta = {r["task_id"]: r for r in map(json.loads, TASK_METADATA.read_text().splitlines()) if r}
    #load the task risk levels
    risk = {r["task_id"]: r for r in map(json.loads, TASK_RISK.read_text().splitlines()) if r}
    #create a dictionary to store the task rows
    rows = {}
    #iterate over the task metadata
    for tid, m in meta.items():
        #if the task id is not in the task risk levels, continue
        if tid not in risk:
            continue
        #create a dictionary to store the task row
        rows[tid] = {
            "tier_ord": TIER_ORD[m["difficulty_tier"]],
            "pages_to_traverse": m["pages_to_traverse"],
            "retrieval_type": m["retrieval_type"],
            "interaction": m["interaction"],
            "target_locatability": m["target_locatability"],
            "risk_level": risk[tid]["risk_level"],
            "site": m["site"],
            "task_category": m["task_category"],
        }
    #print the number of tasks loaded
    print(f"[strategy_c] task features loaded for {len(rows)} tasks")
    #return the task rows
    return rows

#check if the router and tasks are loaded
def _ready() -> tuple[Router, dict[int, dict]]:
    #if the router or tasks are not loaded, configure them
    if _router is None or _tasks is None:
        configure()
    return _router, _tasks


#episode function for the router-directed dynamic strategy
def run_episode(task_id: int, budget: int | None = None) -> dict:
    #get the router and tasks
    router, tasks = _ready()
    #get the budget
    cap = budget if budget is not None else DEFAULT_BUDGET
    #get the site
    site = W.site_of(task_id)
    #get the task row
    task_row = {**tasks[task_id], "budget_level": cap}
    #choose the mode
    mode, p_cycle = router.choose_mode(task_row)
    #print the task id, site, budget, and mode
    print(f"\n{'=' * 60}\nTASK {task_id}  (site: {site})  [C, budget {cap}]  "
          f"router -> {mode} (p_cycle={p_cycle:.3f})\n{'=' * 60}")
 
    #make the environment
    env = W.make_env(task_id)
    #reset the environment
    obs, _ = env.reset()
    #get the goal
    goal = W.goal_of(obs)
    #print the goal and start URL
    print(f"Goal: {goal}\nStart URL: {obs.get('url', 'unknown')}\n")
 
    #initialize the input and output tokens
    in_tok = out_tok = 0
    #create a list to store the steps
    steps: list[dict] = []
    #create a list to store the decisions
    decisions: list[dict] = []
    #create a list to store the action history
    action_history: list[str] = []
    url_history: list[str] = [obs.get("url", "")]
    #create a counter to store the url action counts
    url_action_counts: Counter = Counter()
    #initialize the consecutive errors
    consecutive_errors = 0
    #initialize the critic revisions
    critic_revisions = 0
    #initialize the last error
    last_error = False
    url_changed = False
    #initialize the success
    success = False
    #initialize the reason
    reason = "max_steps"
    reward = 0.0
    #initialize the stop step
    stop_step: int | None = None
    #initialize the time
    t0 = time.time()
 
    for i in range(W.MAX_STEPS):
        #compute the spent tokens
        spent = in_tok + out_tok
        #create the state
        state = {**task_row, "step_index": i,
                 "budget_remaining_frac": 1 - spent / cap,
                 "last_error": int(last_error), "url_changed_last": int(url_changed),
                 "consecutive_errors": consecutive_errors, "mode": mode}
        #compute the probability of a stop
        p_stop = router.p_stop(state)
        #check if the stop is triggered
        stop = p_stop >= router.threshold
        #store the decision
        decisions.append({"step": i, "p_stop": round(p_stop, 6), "stop": stop,
                          "budget_remaining_frac": round(state["budget_remaining_frac"], 4),
                          "last_error": int(last_error), "url_changed_last": int(url_changed),
                          "consecutive_errors": consecutive_errors})
        #print the step, router p_stop, and stop decision
        print(f"\n step {i}   router p_stop={p_stop:.4f} -> {'STOP' if stop else 'continue'}")
        #if the stop is triggered, set the stop step and reason
        if stop:
            stop_step = i
            reason = "router_stop"
            #if the stop answer is "na", send a message to the user
            if STOP_ANSWER == "na":
                obs, reward, terminated, truncated, _ = env.step("send_msg_to_user('N/A')")
                #if the reward is 1.0, set the success to True
                if reward >= 1.0:
                    success = True
            break
 
        #build the base prompt
        base = W.build_prompt(obs, action_history, url_history)
        #get the current url
        cur_url = obs.get("url", "")
 
        #if the mode is "execute", take one website step
        if mode == "execute":
            #compute the estimated input tokens
            est_in = len(base) // 4
            #if the spent tokens plus the estimated input tokens exceeds the budget, set the reason to "budget_would_exceed"
            if spent + est_in > cap:
                reason = "budget_would_exceed"
                print(f" SKIPPED - est. {est_in} input tokens would exceed cap (used {spent}/{cap})")
                break
            #call the agent
            result = W.call_agent(strategy_a.agent, base)
            #get the usage
            u = result.usage()
            #update the input and output tokens
            in_tok += u.input_tokens
            out_tok += u.output_tokens
            #get the final action
            final_action = result.output.action
            #print the reason, action, url, and tokens
            print(f"reason: {result.output.reasoning}\naction: {final_action}\nurl: {cur_url[:90]}")
            print(f"tokens: +{u.input_tokens} in / +{u.output_tokens} out (cumulative {in_tok + out_tok})")
            #store the step
            steps.append({
                "step": i, "agent_role": "single", "action": final_action, "url": cur_url,
                "prompt_chars": len(base), "input_tokens": u.input_tokens,
                "output_tokens": u.output_tokens, "cumulative_tokens": in_tok + out_tok,
                "usage_details": dict(u.details) if getattr(u, "details", None) else None,
                "router_p_stop": round(p_stop, 6),
            })
            #if the input and output tokens exceed the budget, set the reason to "safety_token_cap"
            if in_tok + out_tok >= cap:
                reason = "safety_token_cap"
                break
 
        #if the mode is "cycle", take one website step
        else:
            #compute the estimated step
            #identical to strategy_b.run_episode's step body
            est_step = ((len(base) // 4) * 3 + strategy_b._INSTRUCTION_TOKENS + strategy_b._OUTPUT_MARGIN)
            if spent + est_step > cap:
                reason = "budget_would_exceed"
                print(f" SKIPPED - est. {est_step} tokens for a 3-agent step would exceed "
                      f"cap (used {spent}/{cap})")
                break
            #record function to store the step
            def record(role, usage, action, extra=None):
                nonlocal in_tok, out_tok
                in_tok += usage.input_tokens
                out_tok += usage.output_tokens
                #create a dictionary to store the step
                rec = {"step": i, "agent_role": role, "action": action, "url": cur_url,
                       "prompt_chars": len(base), "input_tokens": usage.input_tokens,
                       "output_tokens": usage.output_tokens,
                       "cumulative_tokens": in_tok + out_tok,
                       "usage_details": dict(usage.details) if getattr(usage, "details", None) else None}
                #if extra is not None, update the step
                if extra:
                    #update the step
                    rec.update(extra)
                #if the role is "planner", update the router p_stop
                if role == "planner":
                    #update the router p_stop
                    rec["router_p_stop"] = round(p_stop, 6)
                steps.append(rec)
                print(f"{role:8s}: +{usage.input_tokens} in / +{usage.output_tokens} out "
                      f"(cumulative {in_tok + out_tok})")
            #call the planner
            r_plan = W.call_agent(strategy_b.planner, base)
            #if the input and output tokens exceed the budget, set the reason to "budget_exhausted_mid_step"
            if in_tok + out_tok >= cap:
                reason = "budget_exhausted_mid_step"
                break
            #get the plan
            plan = r_plan.output.plan
            #store the step
            record("planner", r_plan.usage(), None, {"plan": plan})
            #print the plan
            print(f"plan: {plan}")
            #call the executor
            r_exec = W.call_agent(strategy_b.executor, f"{base}\n\nPLANNER SUB-GOAL:\n{plan}")
            #get the proposed action
            proposed = r_exec.output.action
            #store the step
            record("executor", r_exec.usage(), proposed, {"proposed_action": proposed})
            #if the input and output tokens exceed the budget, set the reason to "budget_exhausted_mid_step"
            if in_tok + out_tok >= cap:
                reason = "budget_exhausted_mid_step"
                break
            #print the proposed action
            print(f"proposed: {proposed}")
            #call the critic
            r_crit = W.call_agent(strategy_b.critic,
                                  f"{base}\n\nPLANNER SUB-GOAL:\n{plan}\n\n"
                                  f"EXECUTOR PROPOSED ACTION:\n{proposed}")
            #get the verdict
            verdict = r_crit.output
            #check if the action is revised
            revised = (not verdict.approve) and bool(verdict.revised_action)
            final_action = verdict.revised_action if revised else proposed
            #update the critic revisions
            critic_revisions += int(revised)
            #store the step
            record("critic", r_crit.usage(), final_action,
                   {"approved": verdict.approve, "revised": revised,
                    "proposed_action": proposed, "critic_reasoning": verdict.reasoning})
            print(f"critic: {'REVISED' if revised else 'approved'} -> {final_action}")
            if in_tok + out_tok >= cap:
                reason = "safety_token_cap"
                break
 
        #update the action history
        action_history.append(final_action)
        #create a pair of the current url and the final action
        pair = (cur_url, final_action)
        #update the url action counts
        url_action_counts[pair] += 1
        #get the number of times the pair has been seen
        pair_n = url_action_counts[pair]
 
        #step the environment
        obs, reward, terminated, truncated, _ = env.step(final_action)
        #get the new url
        new_url = obs.get("url", "")
        #update the url history
        url_history.append(new_url)
        #get the last action error
        err = obs.get("last_action_error")
        #update the consecutive errors
        consecutive_errors = consecutive_errors + 1 if err else 0
        #update the last error
        last_error = bool(err)
        #update the url changed
        url_changed = new_url != cur_url
 
        #update the steps
        for s in steps:
            #if the step index is the current step, update the step
            if s["step"] == i:
                s["action_error"] = str(err) if err else None
                s["step_reward"] = reward
                s["next_url"] = new_url
 
        #print the result
        print(f"result: reward={reward}{'  ERROR: ' + str(err)[:70] if err else '  (action accepted)'}")
 
        #if the reward is 1.0, set the success to True and reason to "success"
        if reward >= 1.0:
            success, reason = True, "success"
            break
        if terminated or truncated:
            reason = "env_terminated"
            break
        if consecutive_errors >= W.MAX_CONSECUTIVE_ERRORS:
            reason = "repeated_action_failure"
            break
        if pair_n >= W.MAX_SAME_ACTION_FROM_PAGE:
            reason = "navigation_cycle"
            break
 
    env.close()
    #get the logical steps
    logical_steps = len({s["step"] for s in steps})
    #create a counter to store the tokens by role
    by_role = Counter()
    #iterate over the steps
    for s in steps:
        #update the tokens by role
        by_role[s["agent_role"]] += s["input_tokens"] + s["output_tokens"]
 
    #create a dictionary to store the record output
    record_out = {
        "strategy": "C",
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
        "mode_chosen": mode,
        "critic_revisions": critic_revisions if mode == "cycle" else None,
        "tokens_by_role": dict(by_role),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "wall_clock_seconds": round(time.time() - t0, 1),
        #create a dictionary to store the router output
        "router": {
            "model_dir": router.model_dir,
            "alpha": router.alpha,
            "stop_threshold": router.threshold,
            "stop_answer": STOP_ANSWER,
            "mode": mode,
            "p_cycle": round(p_cycle, 6),
            "stopped_by_router": stop_step is not None,
            "stop_step": stop_step,
            "decisions": decisions,
        },
        "step_log": steps,
    }

    #print the record output
    print(f"\n --- C / {site} / task {task_id} ---")
    #iterate over the keys
    for k in ("mode_chosen", "success", "termination_reason", "steps", "agent_calls",
              "total_tokens", "wall_clock_seconds"):
        print(f" {k:20s}: {record_out[k]}")
    print(f" {'router':20s}: p_cycle={p_cycle:.3f}  stopped_by_router={stop_step is not None}"
          f"{'' if stop_step is None else f' at step {stop_step}'}")
    return record_out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tasks", nargs="*", type=int)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--router-dir", default=None)
    ap.add_argument("--stop-answer", choices=["none", "na"], default=None)
    ap.add_argument("--out", default="../data/raw/strategy_c_results.jsonl")
    args = ap.parse_args()

    configure(args.router_dir, args.stop_answer)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for tid in args.tasks:
            try:
                rec = run_episode(tid, args.budget)
            except Exception as e:
                import traceback
                traceback.print_exc()
                rec = {"strategy": "C", "task_id": tid, "site": W.site_of(tid), "error": f"{type(e).__name__}: {e}"}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    print(f"\nWrote {len(args.tasks)} episodes to {out_path}")


if __name__ == "__main__":
    main()