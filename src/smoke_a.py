"""
Strategy A smoke test - one task per site (shopping, CMS, forum).

Purpose: prove the environment + agent + token accounting work end to end,
and report the core metrics before any large run.

Reports per episode:
  site, task_id, success, termination_reason, steps,
  input_tokens, output_tokens, total_tokens, wall_clock_seconds

Run:
    python smoke_a.py                # auto-picks one task per site
    python smoke_a.py 27 90 200      # or pass explicit task ids
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()   # <-- MUST come before browsergym import

# Map your WA_* names onto the bare names BrowserGym/WebArena expects
for _wa, _bare in [
    ("WA_SHOPPING", "SHOPPING"),
    ("WA_SHOPPING_ADMIN", "SHOPPING_ADMIN"),
    ("WA_REDDIT", "REDDIT"),
    ("WA_GITLAB", "GITLAB"),
    ("WA_WIKIPEDIA", "WIKIPEDIA"),
    ("WA_MAP", "MAP"),
    ("WA_HOMEPAGE", "HOMEPAGE"),
]:
    if os.getenv(_wa) and not os.getenv(_bare):
        os.environ[_bare] = os.environ[_wa]

import types

import openai

if not hasattr(openai, "error"):
    _err = types.ModuleType("openai.error")
    for _n in ["OpenAIError", "APIError", "RateLimitError", "APIConnectionError",
               "AuthenticationError", "InvalidRequestError",
               "ServiceUnavailableError", "Timeout", "TryAgain"]:
        setattr(_err, _n, type(_n, (Exception,), {}))
    openai.error = _err
    sys.modules["openai.error"] = _err

import gymnasium as gym
import browsergym.webarena  # noqa: F401
from browsergym.utils.obs import flatten_axtree_to_str
from pydantic import BaseModel, Field
from pydantic_ai import Agent


# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
MODEL = "openai:gpt-5-mini"     # switch to "openai:gpt-4o-mini" to compare
MAX_STEPS = 12                  # safety stop: a stuck agent can't loop forever
SAFETY_TOKEN_CAP = 16_000       # safety stop on spend for the smoke test
SITES = ["shopping", "shopping_admin", "reddit"]   # your three sites


# ----------------------------------------------------------------------------
# Agent: structured output means no regex parsing of the model's reply
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
"""

REASONING_EFFORT = "low"   # pinned for reproducibility: hidden reasoning tokens
                           # are billed as output, so this must not drift

try:
    from pydantic_ai.models.openai import (
        OpenAIResponsesModel,
        OpenAIResponsesModelSettings,
    )

    _model = OpenAIResponsesModel(MODEL.split(":", 1)[1])
    _settings = OpenAIResponsesModelSettings(
        openai_reasoning_effort=REASONING_EFFORT
    )
    agent = Agent(
        _model,
        output_type=AgentAction,
        instructions=INSTRUCTIONS,
        model_settings=_settings,
    )
    print(f"[config] reasoning_effort pinned to '{REASONING_EFFORT}'")
except Exception as _e:
    print(f"[config] WARNING: could not pin reasoning_effort ({_e}); "
          f"using provider default")
    agent = Agent(MODEL, output_type=AgentAction, instructions=INSTRUCTIONS)

# Playwright's sync API runs inside an event loop, and agent.run_sync() tries to
# start its own inside it -> "This event loop is already running". Calling the
# agent from a worker thread avoids the clash.
_executor = ThreadPoolExecutor(max_workers=1)


def call_agent(prompt: str):
    return _executor.submit(agent.run_sync, prompt).result()

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def config_dir() -> Path:
    """Locate WebArena's task config files, handling namespace packages."""
    import webarena

    candidates: list[Path] = []

    # Namespace packages have __path__ but no __file__
    for p in list(getattr(webarena, "__path__", [])):
        candidates.append(Path(p))
    if getattr(webarena, "__file__", None):
        candidates.append(Path(webarena.__file__).parent)

    for base in candidates:
        for sub in (base / "config_files", base):
            if (sub / "test.raw.json").exists() or list(sub.glob("[0-9]*.json")):
                return sub

    raise FileNotFoundError(
        "Could not find WebArena config files. Searched: "
        + ", ".join(str(c) for c in candidates)
        + "\nSet CONFIG_DIR manually to the folder containing test.raw.json "
          "(clone https://github.com/web-arena-x/webarena and use its config_files/)."
    )


def load_configs() -> list[dict]:
    d = config_dir()
    files = sorted(d.glob("[0-9]*.json"))
    if files:
        return [json.loads(f.read_text()) for f in files]
    return json.loads((d / "test.raw.json").read_text())


def pick_one_task_per_site() -> dict[str, int]:
    """First single-site task id for each of the three target sites."""
    chosen: dict[str, int] = {}
    for cfg in load_configs():
        sites = cfg.get("sites", [])
        if len(sites) == 1 and sites[0] in SITES and sites[0] not in chosen:
            chosen[sites[0]] = cfg["task_id"]
        if len(chosen) == len(SITES):
            break
    return chosen


def site_of(task_id: int) -> str:
    for cfg in load_configs():
        if cfg.get("task_id") == task_id:
            sites = cfg.get("sites", [])
            return sites[0] if sites else "unknown"
    return "unknown"


def build_prompt(obs: dict) -> str:
    goal = obs.get("goal") or " ".join(
        p.get("text", "") for p in obs.get("goal_object", [])
    )
    axtree = flatten_axtree_to_str(obs["axtree_object"])
    err = obs.get("last_action_error") or "none"
    return (
        f"GOAL:\n{goal}\n\n"
        f"URL: {obs.get('url', 'unknown')}\n"
        f"LAST ACTION ERROR: {err}\n\n"
        f"PAGE (AXTree):\n{axtree}"
    )


# ----------------------------------------------------------------------------
# One episode
# ----------------------------------------------------------------------------
def run_episode(task_id: int) -> dict:
    site = site_of(task_id)
    print(f"\n{'=' * 60}\nTASK {task_id}  (site: {site})\n{'=' * 60}")

    env = gym.make(f"browsergym/webarena.{task_id}", timeout=60000)
    obs, _ = env.reset()

    #Print the task goal and start URL
    goal = obs.get("goal") or " ".join(
        p.get("text", "") for p in obs.get("goal_object", [])
    )
    print(f"Goal: {goal}")
    print(f"Start URL: {obs.get('url', 'unknown')}\n")

    in_tok = out_tok = 0
    steps: list[dict] = []
    success = False
    reason = "max_steps"
    reward = 0.0
    t0 = time.time()

    for i in range(MAX_STEPS):
        result = call_agent(build_prompt(obs))
        u = result.usage()
        in_tok += u.input_tokens
        out_tok += u.output_tokens
        total = in_tok + out_tok
        decided = result.output

        print(f"\n step {i}")
        print(f"   reason : {decided.reasoning}")
        print(f"   action : {decided.action}")
        print(f"   tokens : +{u.input_tokens} in / +{u.output_tokens} out"
              f"   (cumulative {total})")
        if getattr(u, "details", None):
            print(f"   detail : {u.details}")

        steps.append({
            "step": i,
            "action": decided.action,
            "input_tokens": u.input_tokens,
            "output_tokens": u.output_tokens,
            "cumulative_tokens": total,
        })

        if total >= SAFETY_TOKEN_CAP:
            reason = "safety_token_cap"
            break

        obs, reward, terminated, truncated, _ = env.step(decided.action)
        err = obs.get("last_action_error")
        print(f"   result : reward={reward}"
              f"{'  ERROR: ' + str(err) if err else '  (action accepted)'}")

        if reward >= 1.0:
            success, reason = True, "success"
            break
        if terminated or truncated:
            reason = "env_terminated"
            break

    env.close()

    record = {
        "site": site,
        "task_id": task_id,
        "goal": goal,
        "success": success,
        "final_reward": reward,
        "termination_reason": reason,
        "steps": len(steps),
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "total_tokens": in_tok + out_tok,
        "wall_clock_seconds": round(time.time() - t0, 1),
        "step_log": steps,
    }

    print(f"\n --- {site} / task {task_id} ---")
    for k in ("success", "termination_reason", "steps",
              "input_tokens", "output_tokens", "total_tokens",
              "wall_clock_seconds"):
        print(f"   {k:20s}: {record[k]}")
    return record


# ----------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) > 1:
        task_ids = [int(a) for a in sys.argv[1:]]
    else:
        chosen = pick_one_task_per_site()
        print("Auto-selected one task per site:")
        for s, t in chosen.items():
            print(f"   {s:15s} -> task {t}")
        task_ids = list(chosen.values())

    results = []
    for tid in task_ids:
        try:
            results.append(run_episode(tid))
        except Exception as e:                      # keep going if one site fails
            print(f"\n!! task {tid} failed: {type(e).__name__}: {e}")
            results.append({"task_id": tid, "site": site_of(tid),
                            "error": f"{type(e).__name__}: {e}"})

    print(f"\n\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    print(f"{'site':16s}{'task':>6s}{'ok':>6s}{'steps':>7s}"
          f"{'tokens':>9s}{'secs':>8s}  reason")
    for r in results:
        if "error" in r:
            print(f"{r['site']:16s}{r['task_id']:>6}{'ERR':>6s}"
                  f"{'-':>7s}{'-':>9s}{'-':>8s}  {r['error'][:40]}")
        else:
            print(f"{r['site']:16s}{r['task_id']:>6}"
                  f"{str(r['success']):>6s}{r['steps']:>7}"
                  f"{r['total_tokens']:>9}{r['wall_clock_seconds']:>8}"
                  f"  {r['termination_reason']}")
            print(f"    goal: {r.get('goal', '')[:80]}")


    Path("smoke_a_results.json").write_text(json.dumps(results, indent=2))
    print("\nFull per-step log written to smoke_a_results.json")


if __name__ == "__main__":
    main()