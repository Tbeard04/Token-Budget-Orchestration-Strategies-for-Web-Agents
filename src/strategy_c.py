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

