"""
spending_scheme_config.py - the variables behind the four spending schemes
"""

#global budget
BUDGET = 32_000

#even share = BUDGET / STEP_BASIS = 6,400 per step
STEP_BASIS = 5

#four spending schemes (pay as you go is the default batches completed already)
SCHEMES = ["pay_as_you_go", "even", "front_loaded", "reactive"]

#front_loaded: step 0 may use this share of the budget, at high effort with the full page
#the remainder is shared over the other (STEP_BASIS - 1) steps
FRONT_FIRST_SHARE = 0.40

#reactive: start from the even share, then scale by how the last step went
#last move errored, or the page did not change
REACTIVE_STUCK_MULT = 1.5

#last move changed the page without error
REACTIVE_PROGRESS_MULT = 0.75

#a step is never trimmed below this many prompt characters
#agent always sees the goal, URL, error line and some of the page
MIN_PROMPT_CHARS = 1_200

#tokens reserved inside a step's allowance for things other than the page
OUTPUT_MARGIN = {"low": 400, "high": 1_500}

#number of tasks and difficulty tier split
N_TASKS = 100
TIER_SPLIT = {"Easy": 33, "Medium": 33, "Hard": 34}
TASK_SEED = 42