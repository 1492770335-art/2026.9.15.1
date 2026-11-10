# -*- coding: utf-8 -*-
"""EXP-0011: market context on strong-GRU (input 100->300 per tick, same-tick cross-sectional
mean/std appended). Only runs after EXP-0008 shows market value on MLP (premise gate).
Single variable: market context on/off vs strong-GRU (EXP-0005).
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
from torch.nn import functional as F

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
from script.evaluate import Evaluator  # noqa: E402
from src.eval.metrics import pooled_pearson_raw  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0011")
os.makedirs(OUT, exist_ok=True)

CFG = {"hidden": 128, "window": 100, "lr": 1e-4, "weight_decay": 1e-4, "epochs": 30,
       "batch_size": 1024, "seed": 42, "clip": 1.0, "eval_every": 2, "patience": 4, "in_dim": 300}


def load_block(dates):
    lf = pl.scan_parquet(DATA).filter(pl.col("date").is_in(sorted(dates)))
    frame = lf.sort(["date", "time", "code"]).collect()
    feats = sorted([n for n in frame.columns if n.startswith("f")], key=lambda n: int(n[1:]))
    X = frame.select(feats).to_numpy().astype(np.float32)
    y = frame.select(["y0", "y1", "y2"]).to_numpy().astype(np.float32)
    d = frame["date"].to_numpy()
    changed = np.r_[True, d[1:] != d[:-1]]
    day_start_row = np.maximum.accumulate(np.where(changed, np.arange(len(d)), 0))
    tick_in_day = (np.arange(len(d)) // 50 - day_start_row // 50).astype(np.int64)
    return X, y, tick_in_day


def add_market(X):
    T = X.reshape(-1, 50, 100)
    m = T.mean(axis=1, keepdims=True)
    s = T.std(axis=1, keepdims=True, ddof=1)
    m = np.broadcast_to(m, T.shape).reshape(-1, 100)
    s = np.broadcast_to(s, T.shape).reshape(-1, 100)
    return np.concatenate([X, m, s], axis=1).astype(np.float32)


class GRU(nn.Module):
    def __init__(self, hidden=128, in_dim=300):
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.output(h[:, -1])


def make_batch(X, I, tick_in_day, W=100):
    B = len(I)
    l = np.minimum(W, tick_in_day[I] + 1)
    k = np.arange(W)
    src = I[:, None] - (W - 1 - k)[None, :] * 50
    valid = k[None, :] >= (W - l)[:, None]
    src = np.clip(src, 0, len(X) - 1)
    Xg = X[src]
    Xg[~valid] = 0.0
    return Xg


def predict_batched(model, X, tick_in_day, device, chunk=4096):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            I = np.arange(s, min(s + chunk, len(X)))
            Xg = torch.from_numpy(make_batch(X, I, tick_in_day, CFG["window"])).to(device)
            outs.append(model(Xg).cpu().numpy())
    return np.concatenate(outs)


def train_one(X, y, label, Xva, yva, tid_tr, tid_va, device, ckpt_dir, seed):
    torch.manual_seed(seed)
    model = GRU(CFG["hidden"], CFG["in_dim"]).to(device)
    Y = torch.from_numpy(y[:, label:label + 1].astype(np.float32)).to(device)
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
            Xg = torch.from_numpy(make_batch(X, I, tid_tr, CFG["window"])).to(device)
            loss = F.mse_loss(model(Xg), Y[I])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
            opt.step()
            total += loss.detach().item() * len(I)
            cnt += len(I)
        train_loss = total / cnt
        if (epoch + 1) % CFG["eval_every"] == 0:
            p = predict_batched(model, Xva, tid_va, device)
            c = pooled_pearson_raw(p, yva[:, label:label + 1])[0]
            history.append({"epoch": epoch + 1, "train_loss": train_loss, "val_own_corr": float(c)})
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
        "        self.window = config.get('window', 100)\n"
        "        self.history = None\n"
        "    def forward(self, x):\n"
        "        h, _ = self.gru(x)\n"
        "        return self.output(h[:, -1])\n"
        "    def reset(self):\n"
        "        self.history = None\n"
        "    def predict(self, x):\n"
        "        current = x[:, None].detach()\n"
        "        self.history = current if self.history is None else torch.cat([self.history, current], dim=1)\n"
        "        self.history = self.history[:, -self.window:]\n"
        "        history = F.pad(self.history, (0, 0, self.window - self.history.shape[1], 0))\n"
        "        m = history.mean(1, keepdim=True)\n"
        "        s = history.std(1, keepdim=True)\n"
        "        hm = torch.cat([history, m.expand_as(history), s.expand_as(history)], dim=2)\n"
        "        return self.forward(hm)\n"
    )
    with open(os.path.join(commit_dir, "network.py"), "w", encoding="utf-8") as f:
        f.write(net)
    for label in range(3):
        with open(os.path.join(commit_dir, "conf", f"Model_{label}.json"), "w", encoding="utf-8") as f:
            json.dump({"model": "GRU", "hidden": CFG["hidden"], "window": CFG["window"]}, f)
        torch.save(sds[label], os.path.join(commit_dir, "model", f"Model_{label}.pth"))


def run_task(name, train_dates, val_dates, device):
    t0 = time.time()
    Xtr, ytr, tid_tr = load_block(train_dates)
    Xva, yva, tid_va = load_block(val_dates)
    Xtr = add_market(Xtr)
    Xva = add_market(Xva)
    ckpt = os.path.join(OUT, f"ckpt_{name}")
    os.makedirs(ckpt, exist_ok=True)
    sds, bests, hists = [], [], []
    for label in range(3):
        b, h = train_one(Xtr, ytr, label, Xva, yva, tid_tr, tid_va, device, ckpt, CFG["seed"])
        sds.append(torch.load(os.path.join(ckpt, f"Model_{label}.pth"), map_location="cpu"))
        bests.append({k: v for k, v in b.items() if k != "sd"})
        hists.append(h)
    commit_dir = os.path.join(OUT, f"commit_grumkt_{name}")
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
