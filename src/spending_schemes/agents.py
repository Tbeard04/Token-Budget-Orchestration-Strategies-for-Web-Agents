"""
agents.py - the four agents at both reasoning efforts
"""
from __future__ import annotations

from contextlib import contextmanager

import wa_env as W
import strategy_a
import strategy_b

#context manager to temporarily set the reasoning effort to the given level
@contextmanager
def _effort(level: str):
    old = W.REASONING_EFFORT
    W.REASONING_EFFORT = level
    try:
        yield
    finally:
        W.REASONING_EFFORT = old

#build the agents at high effort
with _effort("high"):
    _single_high = W.make_agent(strategy_a.INSTRUCTIONS, strategy_a.AgentAction, label="strategy_a_high")
    _planner_high = W.make_agent(strategy_b.PLANNER_INSTRUCTIONS, strategy_b.PlannerPlan, label="planner_high")
    _executor_high = W.make_agent(strategy_b.EXECUTOR_INSTRUCTIONS, strategy_b.ExecutorAction, label="executor_high")
    _critic_high = W.make_agent(strategy_b.CRITIC_INSTRUCTIONS, strategy_b.CriticVerdict, label="critic_high")

#the four agents at both reasoning efforts
AGENTS = {
    "low": {"single": strategy_a.agent, "planner": strategy_b.planner,
            "executor": strategy_b.executor, "critic": strategy_b.critic},
    "high": {"single": _single_high, "planner": _planner_high,
             "executor": _executor_high, "critic": _critic_high},
}

#instruction overhead per step, in tokens, at the same 4:1 estimate the strategies use
#A's single agent: one instruction block.
#B's cycle: three
INSTRUCTION_TOKENS = {
    "execute": len(strategy_a.INSTRUCTIONS) // 4,
    "cycle": strategy_b._INSTRUCTION_TOKENS,
}