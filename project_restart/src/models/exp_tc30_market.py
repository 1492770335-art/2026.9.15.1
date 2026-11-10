# -*- coding: utf-8 -*-
"""TC-30 M1: K16-GRU + market context (MKT-v1) single-factor ablation, seed42, folds A/B.
Control = EXP-0016 K16 seed42 (A 0.2390 / B 0.2612). Only adds pre-registered market features
(per-feature current-tick cross-sectional mean+std, input 100->300). Same GRU128/lr1e-4/wd1e-4/
batch1024/clip1/seed42/window16, new inner early-stop protocol, official Evaluator.
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
sys.path.insert(0, os.path.join("project_restart", "src", "models"))
from script.evaluate import Evaluator  # noqa: E402
from gru_strong import load_block, make_batch, CFG  # noqa: E402
from src.eval.metrics import pooled_pearson_raw, official_score  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0019")
os.makedirs(OUT, exist_ok=True)

K = 16
CFG["window"] = K
CAP = 30
PATIENCE = 4
EVAL_EVERY = 2
SEED = 42
CONTROL = {"A": 0.2390, "B": 0.2612}
IN_DIM = 300

FOLDS = {
    "A": {"inner_train": [0, 1, 2, 3, 4, 5], "inner_stop": [6, 7],
          "full_train": list(range(0, 8)), "outer_val": [8, 9, 10, 11]},
    "B": {"inner_train": list(range(0, 10)), "inner_stop": [10, 11],
          "full_train": list(range(0, 12)), "outer_val": [12, 13, 14, 15]},
}


def add_market(X):
    T = X.reshape(-1, 50, 100)
    m = T.mean(axis=1, keepdims=True)
    s = T.std(axis=1, keepdims=True, ddof=1)
    m = np.broadcast_to(m, T.shape).reshape(-1, 100)
    s = np.broadcast_to(s, T.shape).reshape(-1, 100)
    return np.concatenate([X, m, s], axis=1).astype(np.float32)


class GRU300(nn.Module):
    def __init__(self, hidden=128, in_dim=300):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.output(h[:, -1])


def train_epoch_fn(model, X, Y, tid, order, opt, device):
    model.train()
    total, cnt = 0.0, 0
    for s in range(0, len(order), CFG["batch_size"]):
        I = order[s:s + CFG["batch_size"]]
        Xg = torch.from_numpy(make_batch(X, I, tid, K)).to(device)
        loss = F.mse_loss(model(Xg), Y[I])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        opt.step()
        total += loss.detach().item() * len(I)
        cnt += len(I)
    return total / cnt


def predict_batched(model, X, tid, device, chunk=8192):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            I = np.arange(s, min(s + chunk, len(X)))
            Xg = torch.from_numpy(make_batch(X, I, tid, K)).to(device)
            outs.append(model(Xg).cpu().numpy())
    return np.concatenate(outs)


def select_epoch(X, y, tid, label, Xst, yst, tid_st, device):
    torch.manual_seed(SEED)
    model = GRU300(CFG["hidden"], IN_DIM).to(device)
    Y = torch.from_numpy(y[:, label:label + 1].astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    n = len(X)
    best = {"epoch": -1, "c": -2.0}
    traj = []
    for epoch in range(CAP):
        rng = np.random.default_rng(SEED + epoch)
        order = rng.permutation(n)
        train_loss = train_epoch_fn(model, X, Y, tid, order, opt, device)
        if (epoch + 1) % EVAL_EVERY == 0:
            p = predict_batched(model, Xst, tid_st, device)
            c = pooled_pearson_raw(p, yst[:, label:label + 1])[0]
            traj.append({"epoch": epoch + 1, "train_loss": train_loss, "inner_stop_corr": float(c)})
            if c > best["c"]:
                best = {"epoch": epoch + 1, "c": float(c)}
            if epoch + 1 - best["epoch"] >= PATIENCE:
                break
    return best["epoch"], traj


def retrain_full(X, y, tid, label, E, device, out_path):
    torch.manual_seed(SEED)
    model = GRU300(CFG["hidden"], IN_DIM).to(device)
    Y = torch.from_numpy(y[:, label:label + 1].astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    n = len(X)
    bpe = math.ceil(n / CFG["batch_size"])
    losses = []
    for epoch in range(E):
        rng = np.random.default_rng(SEED + epoch)
        order = rng.permutation(n)
        losses.append(train_epoch_fn(model, X, Y, tid, order, opt, device))
    torch.save(model.state_dict(), out_path)
    return bpe * E, losses


def build_commit(sds, commit_dir):
    if os.path.exists(commit_dir):
        import shutil
        shutil.rmtree(commit_dir)
    os.makedirs(os.path.join(commit_dir, "conf"))
    os.makedirs(os.path.join(commit_dir, "model"))
    import shutil as _sh
    _sh.copy(os.path.join("project", "project", "script", "base.py"), os.path.join(commit_dir, "base.py"))
    net = (
        "import torch\n"
        "from torch import nn\n"
        "from torch.nn import functional as F\n"
        "from .base import BaseModel\n"
        "class GRU(BaseModel):\n"
        "    def __init__(self, config):\n"
        "        super().__init__(config)\n"
        "        hidden = config['hidden']\n"
        "        self.gru = nn.GRU(300, hidden, batch_first=True)\n"
        "        self.output = nn.Linear(hidden, 1)\n"
        "        self.window = config.get('window', 16)\n"
        "        self.history = None\n"
        "    def forward(self, x):\n"
        "        h, _ = self.gru(x)\n"
        "        return self.output(h[:, -1])\n"
        "    def reset(self):\n"
        "        self.history = None\n"
        "    def predict(self, x):\n"
        "        m = x.mean(0, keepdim=True)\n"
        "        s = x.std(0, keepdim=True)\n"
        "        xm = torch.cat([x, m.expand_as(x), s.expand_as(x)], dim=1)\n"
        "        cur = xm[:, None].detach()\n"
        "        self.history = cur if self.history is None else torch.cat([self.history, cur], dim=1)\n"
        "        self.history = self.history[:, -self.window:]\n"
        "        h = F.pad(self.history, (0, 0, self.window - self.history.shape[1], 0))\n"
        "        return self.forward(h)\n"
    )
    with open(os.path.join(commit_dir, "network.py"), "w", encoding="utf-8") as f:
        f.write(net)
    for label in range(3):
        with open(os.path.join(commit_dir, "conf", f"Model_{label}.json"), "w", encoding="utf-8") as f:
            json.dump({"model": "GRU", "hidden": CFG["hidden"], "window": K}, f)
        torch.save(sds[label], os.path.join(commit_dir, "model", f"Model_{label}.pth"))


def run_fold(fold):
    spec = FOLDS[fold]
    t0 = time.time()
    Xi, yi, tii = load_block(spec["inner_train"])
    Xs, ys, tis = load_block(spec["inner_stop"])
    Xf, yf, tif = load_block(spec["full_train"])
    Xi, Xs, Xf = add_market(Xi), add_market(Xs), add_market(Xf)
    ckpt = os.path.join(OUT, f"ckpt_mkt_{fold}")
    os.makedirs(ckpt, exist_ok=True)
    info = {}
    sds = []
    for label in range(3):
        E, traj = select_epoch(Xi, yi, tii, label, Xs, ys, tis, device)
        out_path = os.path.join(ckpt, f"Model_{label}.pth")
        steps, losses = retrain_full(Xf, yf, tif, label, E, device, out_path)
        sds.append(torch.load(out_path, map_location="cpu", weights_only=True))
        info[f"y{label}"] = {"E": E, "total_opt_steps_full": steps, "inner_traj": traj, "full_train_losses": losses}
        print(f"[mkt {fold} y{label}] E={E} steps={steps}", flush=True)
    commit_dir = os.path.join(OUT, f"commit_mkt_{fold}")
    build_commit(sds, commit_dir)
    ev = Evaluator(DATA, commit_dir, device)
    pooled = ev.evaluate(date=spec["outer_val"])
    daily = {int(d): ev.evaluate(date=d) for d in spec["outer_val"]}
    extra = {}
    if fold == "A":
        extra["no_date8"] = ev.evaluate(date=[9, 10, 11])
    rec = {"fold": fold, "per_target": info, "pooled_official": pooled, "daily": daily, "extra": extra,
           "delta_vs_control": round(pooled["score"] - CONTROL[fold], 4),
           "elapsed_sec": round(time.time() - t0, 1),
           "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1)}
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    return rec


if __name__ == "__main__":
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device, "K:", K, "IN_DIM:", IN_DIM, flush=True)
    results = {}
    for fold in ("A", "B"):
        rec = run_fold(fold)
        results[f"mkt_{fold}"] = rec
        with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(json.dumps({k: rec[k] for k in ("fold", "per_target", "pooled_official", "daily", "extra", "delta_vs_control", "elapsed_sec", "peak_vram_mb")}, indent=2, default=str), flush=True)
    print("done", flush=True)
