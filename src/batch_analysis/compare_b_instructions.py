"""
this script is specifically for comparing the different batch outputs based on the iterations of instructions for strategy B
i.e. 
data/raw/strategy_b_v1_instructions.jsonl
data/raw/strategy_b_v2_instructions.jsonl
data/raw/strategy_b_v3_instructions.jsonl

Since these are most likely contaminted, the comparison will be done by sorting them into sub-categories of tasks (i.e. read only, state change)
"""
