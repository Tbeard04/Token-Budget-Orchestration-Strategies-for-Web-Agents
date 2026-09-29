# Shared configurations for environment & per agent settings.

"""
wa_env.py - shared WebArena/BrowserGym environment layer.

Everything here is used by ALL strategies. Nothing here decides what action to take; that is the strategies' job. Keeping it in one module means Strategy A,
B and C cannot drift apart on observation format, guards or configuration -- which is what makes the cross-strategy comparison valid.

Contents:
    - .env loading and WA_* -> bare env var mapping
    - openai>=1.0 compatibility shims for WebArena's 2023 code
    - task config discovery and loading
    - build_prompt(): observation -> text
    - make_agent(): agent construction with reasoning_effort pinned
    - call_agent(): off-thread agent invocation
    - shared constants (MODEL, MAX_STEPS, SITES, guard thresholds)
"""
from __future__ import annotations

import json
import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# WA_* --> bare names BrowserGym/WebArena expects
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

# openai>=1.0 compatibility
# WebArena's codebase targets openai 0.x. pydantic-ai requires openai 1.x.
# Rather than downgrade, shim the removed surfaces WebArena still references.
import openai

if not hasattr(openai, "error"):
    _err = types.ModuleType("openai.error")
    for _n in ["OpenAIError", "APIError", "RateLimitError", "APIConnectionError",
               "AuthenticationError", "InvalidRequestError",
               "ServiceUnavailableError", "Timeout", "TryAgain"]:
        setattr(_err, _n, type(_n, (Exception,), {}))
    openai.error = _err
    sys.modules["openai.error"] = _err

EVAL_MODEL = "gpt-4o-mini"

# WebArena's evaluators call openai.ChatCompletion.create(), removed in
# openai>=1.0. Rather than patching every import path that references it,
# put a compatible shim on the openai module itself so ALL callers are covered.
if not hasattr(openai, "ChatCompletion"):
    from openai import OpenAI as _OpenAI
    _chat_client = _OpenAI()

    class _FakeChatCompletion:
        @staticmethod
        def create(model=None, messages=None, temperature=1.0,
                   max_tokens=None, top_p=1.0, stop=None, **kwargs):
            resp = _chat_client.chat.completions.create(
                model=EVAL_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stop=stop,
            )
            return {
                "choices": [
                    {"message": {"content": resp.choices[0].message.content}}
                ]
            }

        @staticmethod
        async def acreate(**kwargs):
            return _FakeChatCompletion.create(**kwargs)

    openai.ChatCompletion = _FakeChatCompletion
    print(f"[wa_env] shimmed openai.ChatCompletion (eval model: {EVAL_MODEL})")



#WebArena's fuzzy-match evaluators call openai.ChatCompletion (removed in 1.0) and hardcode gpt-4-1106-preview (retired). Replace with a v1 client.
#this changes the grader model relative to the original paper

def _patch_webarena_openai() -> None:
    try:
        from llms.providers import openai_utils as _ou  # type: ignore
    except Exception as _e:
        print(f"[wa_env] could not patch WebArena openai_utils: {_e}")
        return

    from openai import OpenAI
    _client = OpenAI()

    #v1 chat completion function
    def _v1_chat(messages, model, temperature, max_tokens, top_p, context_length, stop_token=None):
        resp = _client.chat.completions.create(
            model=EVAL_MODEL,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=[stop_token] if stop_token else None,
        )
        return resp.choices[0].message.content

    _ou.generate_from_openai_chat_completion = _v1_chat

    #The evaluator may have imported the function directly, holding its own reference that the module-level patch above does not reach. So I patch it here too.
    for _mod_name in ("evaluation_harness.evaluators",
                      "webarena.evaluation_harness.evaluators"):
        try:
            import importlib
            _eval_mod = importlib.import_module(_mod_name)
            if hasattr(_eval_mod, "generate_from_openai_chat_completion"):
                _eval_mod.generate_from_openai_chat_completion = _v1_chat
                print(f"[wa_env] also patched {_mod_name}")
        except Exception:
            pass

    print(f"[wa_env] patched WebArena openai_utils (eval model: {EVAL_MODEL})")

_patch_webarena_openai()

#browsergym: everything above must already have run
import gymnasium as gym
import browsergym.webarena
from browsergym.utils.obs import flatten_axtree_to_str
from pydantic_ai import Agent


#shared configuration - identical for every strategy
MODEL = "openai:gpt-5-mini"
REASONING_EFFORT = "low"
MAX_STEPS = 25
SITES = ["shopping", "shopping_admin", "reddit"]
BUDGETS = [2000, 4000, 8000, 16000, 32000, 64000]

#Guard thresholds (applied to A, B and C alike) page will not respond to any action
MAX_CONSECUTIVE_ERRORS = 4

#agent is circling
MAX_SAME_ACTION_FROM_PAGE = 3

#prior actions shown in the prompt
ACTION_HISTORY_LEN = 8

#Task configs
def config_dir() -> Path:
    import webarena

    candidates: list[Path] = []
    for p in list(getattr(webarena, "__path__", [])):
        candidates.append(Path(p))
    if getattr(webarena, "__file__", None):
        candidates.append(Path(webarena.__file__).parent)

    for base in candidates:
        for sub in (base / "config_files", base):
            if (sub / "test.raw.json").exists() or list(sub.glob("[0-9]*.json")):
                return sub

    raise FileNotFoundError(
        + ", ".join(str(c) for c in candidates)
    )

#config cache
_CONFIG_CACHE: list[dict] | None = None

#load the configs
def load_configs() -> list[dict]:
    global _CONFIG_CACHE
    if _CONFIG_CACHE is None:
        d = config_dir()
        files = sorted(d.glob("[0-9]*.json"))
        if files:
            _CONFIG_CACHE = [json.loads(f.read_text()) for f in files]
        else:
            _CONFIG_CACHE = json.loads((d / "test.raw.json").read_text())
    return _CONFIG_CACHE

#get the site of a task
def site_of(task_id: int) -> str:
    for cfg in load_configs():
        if cfg.get("task_id") == task_id:
            sites = cfg.get("sites", [])
            return sites[0] if sites else "unknown"
    return "unknown"

#get the single site tasks
def single_site_tasks(sites: list[str] | None = None) -> dict[str, list[int]]:
    sites = sites or SITES
    out: dict[str, list[int]] = {s: [] for s in sites}
    for cfg in load_configs():
        cfg_sites = cfg.get("sites", [])
        if len(cfg_sites) == 1 and cfg_sites[0] in out:
            out[cfg_sites[0]].append(cfg["task_id"])
    return out

#Observation --> prompt
def build_prompt(obs: dict, history: list[str] | None = None, urls: list[str] | None = None) -> str:
    goal = obs.get("goal") or " ".join(p.get("text", "") for p in obs.get("goal_object", []))

    try:
        axtree = flatten_axtree_to_str(
            obs["axtree_object"],
            extra_properties=obs["extra_element_properties"],
            filter_visible_only=True,
            filter_with_bid_only=True,
        )
    except (TypeError, ValueError, KeyError) as _e:
        axtree = flatten_axtree_to_str(obs["axtree_object"])
        print(f"[wa_env] AXTree filtering unavailable ({_e}); using full tree")

    err = obs.get("last_action_error") or "none"
    if err != "none" and "Timeout" in err and "exceeded" in err:
        #A timeout may mean the action landed and only the navigation wait expired, OR that the element genuinely cannot be actioned. If the
        # same action just failed with no page change
        repeated = (
            history is not None and len(history) >= 2
            and history[-1] == history[-2]
            and urls is not None and len(urls) >= 2
            and urls[-1] == urls[-2]
        )
        #if the action has been repeated more than twice and the page has not changed, 
        #set the error to "this exact action has FAILED more than once and the page has not changed. 
        #It will not work. Choose a DIFFERENT element or a different approach and not to rety it"
        if repeated:
            err = ("this exact action has FAILED more than once and the page has "
                   "not changed. It will not work. Choose a DIFFERENT element or "
                   "a different approach - do not retry it.")
        else:
            err = ("previous action timed out waiting for the page to settle - "
                   "it may have succeeded. Check the current page before retrying.")
    hist = "none yet"
    if history:
        rows = []
        for n in range(max(0, len(history) - ACTION_HISTORY_LEN), len(history)):
            u = urls[n] if urls and n < len(urls) else ""
            rows.append(f"  {n}. {history[n]}  -> {u[:70]}")
        hist = "\n".join(rows)

    return (
        f"GOAL:\n{goal}\n\n"
        f"URL: {obs.get('url', 'unknown')}\n"
        f"LAST ACTION ERROR: {err}\n\n"
        f"ACTIONS YOU HAVE ALREADY TAKEN:\n{hist}\n\n"
        f"PAGE (AXTree):\n{axtree}"
    )

#get the goal of a task
def goal_of(obs: dict) -> str:
    return obs.get("goal") or " ".join(
        p.get("text", "") for p in obs.get("goal_object", [])
    )

#Agent construction and invocation
def make_agent(instructions: str, output_type, label: str = "agent") -> Agent:
    try:
        from pydantic_ai.models.openai import (
            OpenAIResponsesModel,
            OpenAIResponsesModelSettings,
        )
        model = OpenAIResponsesModel(MODEL.split(":", 1)[1])
        settings = OpenAIResponsesModelSettings(openai_reasoning_effort=REASONING_EFFORT)
        a = Agent(model, output_type=output_type, instructions=instructions, model_settings=settings)
        print(f"[wa_env] {label}: reasoning_effort='{REASONING_EFFORT}'")
        return a
    except Exception as _e:
        print(f"[wa_env] WARNING: {label} could not pin reasoning_effort ({_e})")
        return Agent(MODEL, output_type=output_type, instructions=instructions)


#Playwright's sync API runs inside an event loop; agent.run_sync() would try to start another inside it. Calling from a worker thread avoids the clash
_executor = ThreadPoolExecutor(max_workers=1)

#call the agent
def call_agent(agent: Agent, prompt: str):
    return _executor.submit(agent.run_sync, prompt).result()

#make the environment
def make_env(task_id: int, timeout: int = 10000):
    return gym.make(f"browsergym/webarena.{task_id}", timeout=timeout)