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


# training
#Advantage weighted policy gradient with entropy bonus
# X:states, A:action index taken, W:exp(advantage/beta) for that action
def train_head(X: np.ndarray, A: np.ndarray, W: np.ndarray, cfg: dict) -> MLP:
    #set the seed
    torch.manual_seed(cfg["seed"])
    #define the MLP
    net = MLP(X.shape[1], cfg["hidden"])
    #define the optimizer
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    #convert the input to tensors
    Xt, At, Wt = (torch.tensor(X, dtype=torch.float32), torch.tensor(A), torch.tensor(W, dtype=torch.float32))
    #train the MLP  
    for epoch in range(cfg["epochs"]):
        perm = torch.randperm(len(Xt))
        for i in range(0, len(Xt), cfg["batch"]):
            j = perm[i:i + cfg["batch"]] #batch index
            logp = torch.log_softmax(net(Xt[j]), -1) #log probability of the action
            chosen = logp.gather(1, At[j][:, None]).squeeze(1) #log probability of the chosen action
            entropy = -(logp.exp() * logp).sum(-1).mean() #entropy of the policy
            loss = -(Wt[j] * chosen).mean() - cfg["alpha"] * entropy #loss function
            opt.zero_grad() #zero the gradients
            loss.backward() #backpropagate the loss
            opt.step() #update the parameters
    return net

#predict the action probability
def predict(net: MLP, X: np.ndarray) -> np.ndarray:
    #disable gradient computation
    with torch.no_grad():
        #predict the action probability
        return torch.softmax(net(torch.tensor(X, dtype=torch.float32)), -1)[:, 1].numpy()


#returns(n, 2) return of each action per state --> (n, 2) weights
#Standard AWR weight is exp(A/beta)
#advantage weights are clipped to avoid overflow
def advantage_weights(returns: np.ndarray, beta: float, clip: float = 20.0) -> np.ndarray:
    #calculate the advantage
    adv = returns - returns.mean(1, keepdims=True)
    #calculate the weights
    w = np.minimum(np.exp(adv / beta), clip)
    #subtract the minimum weight to avoid negative weights
    return w - w.min(1, keepdims=True)


#build the mode set
def build_mode_set(pairs: list[dict], lam: float):
    #filter the pairs that are both eligible
    P = [p for p in pairs if p["both_eligible"]]
    #convert the pairs to features
    X = np.array([task_features(p) for p in P], dtype=np.float32)
    #convert the pairs to returns
    R = np.array([[p["a_success"] * (1 - lam * p["a_tokens"] / p["budget_level"]),
                   p["b_success"] * (1 - lam * p["b_tokens"] / p["budget_level"])] for p in P])
    #return the pairs, features, and returns
    return P, X, R


#build the stop set
def build_stop_set(trans: list[dict], lam: float):
    #filter the transitions that are not terminal and are reward eligible
    S = [t for t in trans if not t["is_terminal"] and t["use_for_reward"]]
    #convert the transitions to features
    X = np.array([stop_features(t) for t in S], dtype=np.float32)
    #convert the transitions to returns
    R = np.zeros((len(S), 2))
    for i, t in enumerate(S):
        #calculate the spent tokens
        spent = (1 - max(0.0, t["budget_remaining_frac"])) * t["budget_level"]
        #calculate the remaining tokens
        remaining = max(0.0, t["episode_tokens"] - spent)
        #calculate the return
        R[i, 0] = float(t["episode_success"]) - lam * remaining / t["budget_level"]
    #return the transitions, features, and returns
    return S, X, R


#expand the features and returns to training rows
def expand(X: np.ndarray, R: np.ndarray, beta: float):
    #calculate the weights
    W = advantage_weights(R, beta)
    #number of states
    n = len(X)
    #concatenate the features and actions
    Xe = np.concatenate([X, X])
    Ae = np.concatenate([np.zeros(n, int), np.ones(n, int)])
    We = np.concatenate([W[:, 0], W[:, 1]]) #concatenate the weights
    keep = We > 0 #keep the rows with weight greater than 0
    #return the features, actions, and weights
    return Xe[keep], Ae[keep], We[keep]


#evaluate the policy
#run the policy on held-out pairs using logged episodes
def simulate(pairs: list[dict], p_cycle: dict, p_stop: dict, threshold: float, episodes_by_arm: dict) -> dict:
    #number of successes
    succ = tok = 0
    #number of stops at each step
    stops_at = Counter()
    #number of modes
    modes = Counter()
    for p in pairs:
        key = (p["task_id"], p["budget_level"]) #task id and budget level
        arm = "B" if p_cycle[key] >= 0.5 else "A"
        modes[arm] += 1 #increment the mode count
        ep = episodes_by_arm[(arm,) + key] #get the episodes by arm and key
        probs = p_stop.get((arm,) + key, []) #get the stop probabilities by arm and key
        stopped = next((k for k, q in enumerate(probs) if q >= threshold), None) #get the stopped step
        if stopped is None:
            succ += ep["success"] #increment the success count
            tok += ep["tokens"] #increment the token count
            stops_at["never"] += 1
        else:
            #increment the token count
            tok += ep["spent"][stopped]
            stops_at["step0" if stopped == 0 else "mid"] += 1
    return {"successes": int(succ), "tokens": int(tok),
            "mode_mix": dict(modes), "stops": dict(stops_at)}
#baselines
def baselines(pairs: list[dict]) -> dict:
    #return the successes and tokens for each arm in (("always_A", "a"), ("always_B", "b"))
    out = {}
    #always choose arm A
    for name, arm in (("always_A", "a"), ("always_B", "b")):
        out[name] = {"successes": int(sum(p[f"{arm}_success"] for p in pairs)),
                     "tokens": int(sum(p[f"{arm}_tokens"] for p in pairs))}
    #oracle
    s = t = 0
    #calculate the successes and tokens for the oracle
    for p in pairs:
        wins = [p[f"{a}_tokens"] for a in "ab" if p[f"{a}_success"]] #get the tokens for the winning arm
        if wins:
            s += 1 #increment the success count
            t += min(wins) #increment the token count
            t += min(wins)
    out["oracle"] = {"successes": s, "tokens": t}
    return out


#load the JSONL file
def load_jsonl(path: str | Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
 
#run the router
def run(pairs: list[dict], trans: list[dict], cfg: dict, out_dir: Path) -> dict:
    t0 = time.time()
    np.random.seed(cfg["seed"])
    lam, beta = cfg["lambda"], cfg["beta"]
 
    P, Xm, Rm = build_mode_set(pairs, lam)
    S, Xs, Rs = build_stop_set(trans, lam)
    groups_m = np.array([p["task_id"] for p in P])
    groups_s = np.array([t["task_id"] for t in S])
    #task ids
    task_ids = sorted({p["task_id"] for p in pairs} | {t["task_id"] for t in trans})
    perm = np.random.default_rng(cfg["seed"]).permutation(task_ids) #shuffle the task ids
    #fold of the task ids
    fold_of = {int(t): i % cfg["folds"] for i, t in enumerate(perm)}
 
    #out-of-fold predictions, both heads share the task folds
    oof_cycle = np.zeros(len(P))
    oof_stop = np.zeros(len(S))
    for f in range(cfg["folds"]):
        #training data for the mode head
        tr_m = np.array([fold_of[g] != f for g in groups_m])
        tr_s = np.array([fold_of[g] != f for g in groups_s])
        nm, ns = Norm(Xm[tr_m]), Norm(Xs[tr_s]) #normalise the features
        Xe, Ae, We = expand(nm(Xm[tr_m]), Rm[tr_m], beta)
        net_m = train_head(Xe, Ae, We, cfg) #train the mode head
        oof_cycle[~tr_m] = predict(net_m, nm(Xm[~tr_m]))
        Xe, Ae, We = expand(ns(Xs[tr_s]), Rs[tr_s], beta) #expand the features and returns
        net_s = train_head(Xe, Ae, We, cfg)
        oof_stop[~tr_s] = predict(net_s, ns(Xs[~tr_s]))
 
    #metrics for the mode head
    p_cycle = {(p["task_id"], p["budget_level"]): float(q) for p, q in zip(P, oof_cycle)}
    better_b = np.array([Rm[i, 1] > Rm[i, 0] for i in range(len(P))]) #better arm
    differ = np.array([p["a_success"] != p["b_success"] for p in P]) #differ in success
    mode_metrics = {
        "n_pairs": len(P), #number of pairs
        "picks_cycle_frac": float((oof_cycle >= 0.5).mean()), #fraction of pairs that pick the cycle arm
        "auc_better_arm": float(roc_auc_score(better_b, oof_cycle)) if better_b.any() and not better_b.all() else None, #AUC of the better arm
        "auc_where_outcomes_differ": float(roc_auc_score(
            [p["b_success"] for p, d in zip(P, differ) if d],
            [q for q, d in zip(oof_cycle, differ) if d])) if differ.sum() > 1 else None, #AUC of the where outcomes differ
        "picks_cycle_by_budget": {
            str(b): float(np.mean([q >= 0.5 for p, q in zip(P, oof_cycle) if p["budget_level"] == b]))
            for b in sorted({p["budget_level"] for p in P})}, #fraction of pairs that pick the cycle arm by budget
    }
 
    #metrics for the stop head
    y_succ = np.array([t["episode_success"] for t in S])
    stop_metrics = {
        "n_states": len(S),
        "auc_stop_vs_eventual_failure": float(roc_auc_score(~y_succ, oof_stop)),
        "mean_p_stop_by_tier": {
            tier: float(np.mean([q for t, q in zip(S, oof_stop) if t["difficulty_tier"] == tier]))
            for tier in ["Easy", "Medium", "Hard"]},
    }
 
    #simulate the full policy on held-out pairs
    p_stop = defaultdict(list) #stop probabilities by strategy, task id, and budget level
    for t, q in zip(S, oof_stop):
        p_stop[(t["strategy"], t["task_id"], t["budget_level"])].append(float(q)) #append the stop probability
    eps = {} #episodes by strategy, task id, and budget level
    for t in trans:
        k = (t["strategy"], t["task_id"], t["budget_level"]) #key
        e = eps.setdefault(k, {"success": int(t["episode_success"]), #success
                               "tokens": t["episode_tokens"], "spent": []}) #tokens and spent
        if not t["is_terminal"]: #if not terminal
            e["spent"].append(int((1 - max(0.0, t["budget_remaining_frac"])) * t["budget_level"])) #append the spent
    base = baselines(P) #baselines
    frontier = [{"threshold": None, "quantile": None, **simulate(P, p_cycle, {}, 1.1, eps)}]
    for q in cfg["threshold_quantiles"]: #quantiles of the held-out P(stop)
        th = float(np.quantile(oof_stop, q))
        frontier.append({"threshold": th, "quantile": q,
                         **simulate(P, p_cycle, p_stop, th, eps)})
    ref_s, ref_t = base["always_A"]["successes"], base["always_A"]["tokens"]
    #frontier
    for row in frontier:
        row["success_kept_vs_A"] = round(row["successes"] / ref_s, 3) if ref_s else None
        row["tokens_saved_vs_A"] = round(1 - row["tokens"] / ref_t, 3) if ref_t else None
    ok = [r for r in frontier if r["threshold"] is not None
          and r["success_kept_vs_A"] is not None
          and r["success_kept_vs_A"] >= cfg["min_success_kept"]]
    suggested = max(ok, key=lambda r: r["tokens_saved_vs_A"])["threshold"] if ok else None
 
    per_budget = {}
    for b in sorted({p["budget_level"] for p in P}):
        sub = [p for p in P if p["budget_level"] == b]
        per_budget[str(b)] = {"baselines": baselines(sub), "router": simulate(sub, p_cycle, p_stop, suggested or 1.1, eps)}
 
    #final fit on everything, saved for the live runner
    nm, ns = Norm(Xm), Norm(Xs) #normalise the features
    Xe, Ae, We = expand(nm(Xm), Rm, beta) #expand the features and returns
    net_m = train_head(Xe, Ae, We, cfg) #train the mode head
    Xe, Ae, We = expand(ns(Xs), Rs, beta) #expand the features and returns
    net_s = train_head(Xe, Ae, We, cfg) #train the stop head
 
    out_dir.mkdir(parents=True, exist_ok=True)
    #save the model
    torch.save({
        "config": cfg,
        "mode_features": MODE_FEATURE_NAMES, "stop_features": STOP_FEATURE_NAMES,
        "mode_norm": {"mean": nm.mean.tolist(), "std": nm.std.tolist()},
        "stop_norm": {"mean": ns.mean.tolist(), "std": ns.std.tolist()},
        "mode_state": net_m.state_dict(), "stop_state": net_s.state_dict(),
        "stop_threshold": suggested,
        "sites": SITES, "categories": CATEGORIES,
    }, out_dir / "router.pt")
 
    metrics = {
        "config": cfg,
        "train_seconds": round(time.time() - t0, 1),
        "mode_head": mode_metrics,
        "stop_head": stop_metrics,
        "baselines": base,
        "frontier": frontier,
        "suggested_threshold": suggested,
        "per_budget": per_budget,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    #save the out-of-fold predictions
    with (out_dir / "oof_predictions.jsonl").open("w") as f:
        for p, q in zip(P, oof_cycle):
            k = (p["task_id"], p["budget_level"])
            f.write(json.dumps({
                "task_id": k[0], "budget_level": k[1], "difficulty_tier": p["difficulty_tier"],
                "fold": fold_of[k[0]], "p_cycle": round(float(q), 4),
                "p_stop_A": [round(x, 4) for x in p_stop.get(("A",) + k, [])],
                "p_stop_B": [round(x, 4) for x in p_stop.get(("B",) + k, [])],
                "a_success": p["a_success"], "a_tokens": p["a_tokens"],
                "b_success": p["b_success"], "b_tokens": p["b_tokens"],
            }) + "\n")

    return metrics

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=int)
    ap.add_argument("--alpha", type=float)
    ap.add_argument("--pairs", default="../data/processed/strategy_c_models/mode_pairs.jsonl")
    ap.add_argument("--transitions", default="../data/processed/strategy_c_models/transitions.jsonl")
    args = ap.parse_args()
 
    #output directory
    out_dir = Path(__file__).parent / f"model_{args.model}"
    cfg_path = out_dir / "config.json"
    #configuration
    cfg = dict(DEFAULT_CONFIG)
    #update the configuration
    if cfg_path.exists():
        cfg.update(json.loads(cfg_path.read_text()))
    if args.alpha is not None:
        cfg["alpha"] = args.alpha
    #the sweep grid is an evaluation setting, never persisted per model
    cfg["threshold_quantiles"] = DEFAULT_CONFIG["threshold_quantiles"]
    out_dir.mkdir(parents=True, exist_ok=True)
    #save the configuration
    cfg_path.write_text(json.dumps({k: v for k, v in cfg.items() if k != "threshold_quantiles"}, indent=2))
    #load the pairs and transitions
    pairs = load_jsonl(args.pairs)
    trans = load_jsonl(args.transitions)
    print(f"[router] model_{args.model}: {len(pairs)} pairs, {len(trans)} transitions")
    #run the router
    run(pairs, trans, cfg, out_dir)


if __name__ == "__main__":
    main()