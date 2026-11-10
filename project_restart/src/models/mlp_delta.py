# -*- coding: utf-8 -*-
"""EXP-0014: delta-head MLP with autoregressive level reconstruction.
Train: target delta d_t = y_t - y_{t-1} (y_{-1}=0 at day start), MSE, strong-MLP recipe.
Inference: level_t = level_{t-1} + d_hat_t, reset to None (0) at day start.
Early stop: pooled Pearson of reconstructed levels on val (own target). Official Evaluator final.
"""
import json
import os
import shutil
import sys
import time

import numpy as np
import polars as pl
import torch
from torch import nn

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
from script.evaluate import Evaluator  # noqa: E402
from src.eval.metrics import pooled_pearson_raw  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0014")
os.makedirs(OUT, exist_ok=True)

CFG = {"hidden": 128, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 40, "batch_size": 8192,
       "seed": 42, "eval_every": 2, "patience": 6}


def load_block(dates):
    lf = pl.scan_parquet(DATA).filter(pl.col("date").is_in(sorted(dates)))
    frame = lf.sort(["date", "time", "code"]).collect()
    feats = sorted([n for n in frame.columns if n.startswith("f")], key=lambda n: int(n[1:]))
    X = frame.select(feats).to_numpy().astype(np.float32)
    y = frame.select(["y0", "y1", "y2"]).to_numpy().astype(np.float32)
    d = frame["date"].to_numpy()
    return X, y, d


def make_delta(y, dates):
    """delta per (date,time,code) ordered rows; y_{-1}=0 at each day start."""
    dy = y.copy()
    idx = np.arange(len(y))
    ok = idx >= 50
    ok &= dates == dates[np.clip(idx - 50, 0, len(y) - 1)]
    prev = np.clip(idx - 50, 0, len(y) - 1)
    dy[ok] = y[ok] - y[prev[ok]]
    return dy


def reconstruct_levels(delta, dates):
    """Level reconstruction: per (date, code) series (rows ordered date,time,code):
    lev[tick t, code c] = sum of deltas of (date,code) series up to tick t.
    Uses (T, 50) tick-major view: column c IS the strided series of code c."""
    delta = np.asarray(delta)
    n = len(delta)
    L = delta.reshape(-1, 50).astype(np.float64)  # (T, 50), rows = ticks
    T = L.shape[0]
    tick_day = dates.reshape(-1, 50)[:, 0]        # date of each tick row
    changed = np.r_[True, tick_day[1:] != tick_day[:-1]]
    day_tick_start = np.maximum.accumulate(np.where(changed, np.arange(T), 0))
    cs = np.cumsum(L, axis=0)
    off = np.zeros_like(L)
    valid = day_tick_start > 0
    off[valid] = cs[day_tick_start[valid] - 1]
    return (cs - off).reshape(-1)


class MLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(100, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.network(x)


def predict_batched_delta(model, X, device, chunk=1 << 18):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            outs.append(model(torch.from_numpy(X[s:s + chunk]).to(device)).cpu().numpy())
    return np.concatenate(outs)


def train_one(Xtr, ytr, dtr, label, Xva, yva, dva, device, ckpt_dir, seed):
    torch.manual_seed(seed)
    model = MLP(CFG["hidden"]).to(device)
    X = torch.from_numpy(Xtr).to(device)
    D = torch.from_numpy(dtr[:, label:label + 1].astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    n = X.shape[0]
    best = {"epoch": -1, "score": -2.0, "sd": None}
    history = []
    for epoch in range(CFG["epochs"]):
        model.train()
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(n)
        total, cnt = 0.0, 0
        for s in range(0, n, CFG["batch_size"]):
            I = order[s:s + CFG["batch_size"]]
            loss = nn.functional.mse_loss(model(X[I]), D[I])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.detach().item() * len(I)
            cnt += len(I)
        train_loss = total / cnt
        if (epoch + 1) % CFG["eval_every"] == 0:
            dhat = predict_batched_delta(model, Xva, device)
            lev = reconstruct_levels(dhat[:, 0], dva)
            c = pooled_pearson_raw(lev[:, None], yva[:, label:label + 1])[0]
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_level_corr": float(c)})
            if c > best["score"]:
                best = {"epoch": epoch + 1, "score": float(c), "sd": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
            if epoch + 1 - best["epoch"] >= CFG["patience"]:
                break
    torch.save(best["sd"], os.path.join(ckpt_dir, f"Model_{label}.pth"))
    return best, history


def build_commit(sds, commit_dir):
    if os.path.exists(commit_dir):
        shutil.rmtree(commit_dir)
    os.makedirs(os.path.join(commit_dir, "conf"))
    os.makedirs(os.path.join(commit_dir, "model"))
    shutil.copy(os.path.join("project", "project", "script", "base.py"), os.path.join(commit_dir, "base.py"))
    net = (
        "from torch import nn\n"
        "from .base import BaseModel\n"
        "class MLP(BaseModel):\n"
        "    def __init__(self, config):\n"
        "        super().__init__(config)\n"
        "        hidden = config['hidden']\n"
        "        self.network = nn.Sequential(nn.Linear(100, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))\n"
        "        self.prev = None\n"
        "    def forward(self, x):\n"
        "        return self.network(x)\n"
        "    def reset(self):\n"
        "        self.prev = None\n"
        "    def predict(self, x):\n"
        "        d = self.forward(x)\n"
        "        if self.prev is None:\n"
        "            self.prev = d\n"
        "        else:\n"
        "            self.prev = self.prev + d\n"
        "        return self.prev\n"
    )
    with open(os.path.join(commit_dir, "network.py"), "w", encoding="utf-8") as f:
        f.write(net)
    for label in range(3):
        with open(os.path.join(commit_dir, "conf", f"Model_{label}.json"), "w", encoding="utf-8") as f:
            json.dump({"model": "MLP", "hidden": CFG["hidden"]}, f)
        torch.save(sds[label], os.path.join(commit_dir, "model", f"Model_{label}.pth"))


def run_task(name, train_dates, val_dates, device):
    t0 = time.time()
    Xtr, ytr, dtr = load_block(train_dates)
    Xva, yva, dva = load_block(val_dates)
    ckpt = os.path.join(OUT, f"ckpt_{name}")
    os.makedirs(ckpt, exist_ok=True)
    sds, bests, hists = [], [], []
    for label in range(3):
        b, h = train_one(Xtr, ytr, dtr, label, Xva, yva, dva, device, ckpt, CFG["seed"])
        sds.append(torch.load(os.path.join(ckpt, f"Model_{label}.pth"), map_location="cpu"))
        bests.append({k: v for k, v in b.items() if k != "sd"})
        hists.append(h)
    commit_dir = os.path.join(OUT, f"commit_delta_{name}")
    build_commit(sds, commit_dir)
    ev = Evaluator(DATA, commit_dir, "cuda:0")
    pooled = ev.evaluate(date=val_dates)
    daily = {d: ev.evaluate(date=d) for d in val_dates}
    return {"name": name, "pooled": pooled, "daily": daily, "best": bests, "history": hists,
            "elapsed_sec": round(time.time() - t0, 1),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1)}


if __name__ == "__main__":
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tasks = {
        "foldA": (list(range(0, 8)), [8, 9, 10, 11]),
        "foldB": (list(range(0, 12)), [12, 13, 14, 15]),
    }
    results = {}
    for name, (tr, va) in tasks.items():
        print("running", name, flush=True)
        results[name] = run_task(name, tr, va, device)
        print(json.dumps({k: results[name][k] for k in ("pooled", "daily", "best", "elapsed_sec", "peak_vram_mb")}, indent=2), flush=True)
    with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("done", flush=True)
