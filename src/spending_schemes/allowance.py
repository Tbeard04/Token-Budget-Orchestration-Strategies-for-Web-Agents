"""
allowance.py - what one site/task step may spend under a given scheme
"""

from __future__ import annotations

from spending_schemes import spending_scheme_config as C

#return (tokens this step may use, reasoning effort, share of page to show)
def allowance(scheme: str, step: int, budget: int, spent: int, last_error: bool, url_changed: bool) -> tuple[int, str, float]:
    #remaining budget
    remaining = max(0, budget - spent)
    #even share
    even = budget // C.STEP_BASIS
    #pay as you go
    if scheme == "pay_as_you_go":
        return remaining, "low", 1.0
    #even share
    if scheme == "even":
        # equal slices for STEP_BASIS steps; after that, whatever is left
        return min(even, remaining), "low", 1.0
    #front loaded
    if scheme == "front_loaded":
        if step == 0:
            return min(int(budget * C.FRONT_FIRST_SHARE), remaining), "high", 1.0
        later = int(budget * (1 - C.FRONT_FIRST_SHARE)) // (C.STEP_BASIS - 1)
        return min(later, remaining), "low", 1.0
    #reactive
    if scheme == "reactive":
        if step == 0:
            return min(even, remaining), "low", 1.0
        if last_error or not url_changed:
            #last move errored, or the page did not change
            return min(int(even * C.REACTIVE_STUCK_MULT), remaining), "high", 1.0
        #last move changed the page without error
        return min(int(even * C.REACTIVE_PROGRESS_MULT), remaining), "low", 1.0

    raise ValueError(f"unknown scheme {scheme!r}")

#how many prompt characters fit inside a step allowance
def prompt_char_limit(step_tokens: int, calls_per_step: int, instruction_tokens: int, effort: str) -> int:
    #per call output
    per_call_output = C.OUTPUT_MARGIN[effort]
    #usable tokens
    usable = step_tokens - instruction_tokens - calls_per_step * per_call_output
    #per call prompt tokens
    per_call_prompt_tokens = max(0, usable) // calls_per_step
    #min prompt characters
    return max(C.MIN_PROMPT_CHARS, per_call_prompt_tokens * 4)


#cut the prompt to max_chars
#the page (AXTree) is the last section, so a tail cut removes page content and keeps goal, URL, error and history
def trim_prompt(prompt: str, max_chars: int) -> tuple[str, bool]:
    #if the prompt is already short enough, return it
    if len(prompt) <= max_chars:
        return prompt, False
    #marker to indicate the page was truncated
    marker = "\n[...page truncated to fit this step's allowance...]"
    #return the prompt truncated to max_chars, and True if the page was truncated
    return prompt[: max(0, max_chars - len(marker))] + marker, True