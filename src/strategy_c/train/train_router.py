"""
train_router.py - train and evaluate one version of the Strategy C router


Two small MLPs, trained separately, saved together as router.pt:
J = E[ w(s,a) * log pi(a|s) ] + alpha * H(pi)
w(s,a) = exp(A(s,a) / beta), clipped
R_stop = success - lambda * tokens_remaining / budget   [[(continue) vs 0 (stop)]]
R_mode = success * (1 - lambda * tokens / budget)  [[failure = 0, see build_mode_set]]
"""

from __future__ import annotations
 
import argparse
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path
 
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

#default configuration
DEFAULT_CONFIG = {
    #entropy weight the hyperparameter under test
    "alpha": 0.05,
    #advantage temperature
    "beta": 0.5,
    #token cost weight in the return
    "lambda": 0.5,
    #hidden layer size
    "hidden": 64,
    #number of epochs
    "epochs": 60,
    #learning rate
    "lr": 1e-3,
    #batch size
    "batch": 256,
    #number of folds
    "folds": 5,
    #seed
    "seed": 0,
    #frontier sweep: stop the most hopeless % of states. Quantiles of the held-out P(stop), so the sweep is by rank
    "threshold_quantiles": [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97, 0.98, 0.99],
    #used to pick the suggested threshold
    "min_success_kept": 0.90,
}
 
#sites
SITES = ["reddit", "shopping", "shopping_admin"]
#categories
CATEGORIES = ["bulk_action", "create", "delete", "information_retrieval", "modify_value", "navigation", "other", "purchase"]
#risk order
RISK_ORD = {"read_only": 0, "idempotent": 1, "non_idempotent": 2}
#mode order
MODE_ORD = {"execute": 0, "cycle": 1}
#mode actions
MODE_ACTIONS = ["execute", "cycle"]
#stop actions
STOP_ACTIONS = ["continue", "stop"]


# features - shared with the live runner
def task_features(r: dict) -> list[float]:
    return ([float(r["tier_ord"]), float(r["pages_to_traverse"]),
             float(r["retrieval_type"]), float(r["interaction"]),
             float(r["target_locatability"]), float(RISK_ORD[r["risk_level"]]),
             math.log2(r["budget_level"])]
            + [1.0 if r["site"] == s else 0.0 for s in SITES]
            + [1.0 if r["task_category"] == c else 0.0 for c in CATEGORIES])
 
 
def stop_features(r: dict) -> list[float]:
    return task_features(r) + [
        float(r["step_index"]),
        max(0.0, float(r["budget_remaining_frac"])),
        float(r["last_error"]),
        float(r["url_changed_last"]),
        min(4.0, float(r["consecutive_errors"])),
        float(MODE_ORD[r["mode"]]),
    ]

#mode feature names
MODE_FEATURE_NAMES = (["tier_ord", "pages_to_traverse", "retrieval_type", "interaction", "target_locatability", "risk_ord", "log2_budget"]
                      + [f"site={s}" for s in SITES] + [f"cat={c}" for c in CATEGORIES])
#stop feature names
STOP_FEATURE_NAMES = MODE_FEATURE_NAMES + ["step_index", "budget_remaining_frac", "last_error", "url_changed_last", "consecutive_errors", "mode_ord"]

#multi-layer perceptron
class MLP(nn.Module):
    #number of input features, number of hidden layers, number of output classes
    def __init__(self, n_in: int, hidden: int, n_out: int = 2):
        #initialise the MLP
        super().__init__()
        #define the layers
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, n_out))
    #forward pass
    def forward(self, x):
        return self.net(x)

#normalisation
class Norm:
    #fit the normalisation on the training fold only
    def __init__(self, X: np.ndarray):
        self.mean = X.mean(0)
        self.std = X.std(0) + 1e-6
    #normalise the input
    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.std