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