# -*- coding: utf-8 -*-
"""EXP-0003: official-as-is GRU (hidden=128, window=100) on A/B folds + gap1 sensitivity.
Semantics exactly replicate example/dataset.py __getitem__ (per-stock causal window within day,
left zero-pad to window=100, day-boundary reset), vectorized for GPU.
Recipe: example/conf/GRU/Model_0.json -> AdamW lr=1e-5 wd=1e-4, grad clip 1.0, MSE, epochs=2,
batch=1024, shuffle=False, seed=42. Validation via OFFICIAL Evaluator (cuda).
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

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0003")
os.makedirs(OUT, exist_ok=True)

CFG = {"hidden": 128, "window": 100, "lr": 1e-5, "weight_decay": 1e-4, "epochs": 2,
       "batch_size": 1024, "seed": 42, "clip": 1.0}


def load_block(dates):
    lf = pl.scan_parquet(DATA).filter(pl.col("date").is_in(sorted(dates)))
    frame = lf.sort(["date", "time", "code"]).collect()
    feats = sorted([n for n in frame.columns if n.startswith("f")], key=lambda n: int(n[1:]))
    X = frame.select(feats).to_numpy().astype(np.float32)
    y = frame.select(["y0", "y1", "y2"]).to_numpy().astype(np.float32)
    d = frame["date"].to_numpy()
    # day-start row indices
    changed = np.r_[True, d[1:] != d[:-1]]
    day_start_row = np.maximum.accumulate(np.where(changed, np.arange(len(d)), 0))
    tick_in_day = np.arange(len(d)) // 50 - day_start_row // 50  # ticks since day start for this stock
    return X, y, tick_in_day.astype(np.int64)


class GRU(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.gru = nn.GRU(100, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.output(h[:, -1])


def make_batch(X, I, tick_in_day, W=100):
    """Vectorized equivalent of example/dataset.py __getitem__ for indices I (B,)."""
    B = len(I)
    l = np.minimum(W, tick_in_day[I] + 1)  # (B,)
    k = np.arange(W)  # (W,)
    src = I[:, None] - (W - 1 - k)[None, :] * 50  # (B, W); valid when k >= W-l
    valid = k[None, :] >= (W - l)[:, None]        # (B, W)
    src = np.clip(src, 0, len(X) - 1)
    Xg = X[src]                                   # (B, W, 100)
    Xg[~valid] = 0.0
    return Xg


def train_one(X, y, label, tick_in_day, device, ckpt_dir):
    torch.manual_seed(CFG["seed"])
    model = GRU(CFG["hidden"]).to(device)
    Y = torch.from_numpy(y[:, label:label + 1].astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    n = X.shape[0]
    losses = []
    order = np.arange(n)
    for epoch in range(CFG["epochs"]):
        total, cnt = 0.0, 0
        for s in range(0, n, CFG["batch_size"]):
            I = order[s:s + CFG["batch_size"]]
            Xg = torch.from_numpy(make_batch(X, I, tick_in_day)).to(device)
            yb = Y[I]
            loss = F.mse_loss(model(Xg), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
            opt.step()
            total += loss.detach().item() * len(I)
            cnt += len(I)
        losses.append(total / cnt)
    torch.save(model.state_dict(), os.path.join(ckpt_dir, f"Model_{label}.pth"))
    return losses


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
        "        self.gru = nn.GRU(100, hidden, batch_first=True)\n"
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
        "        return self.forward(history)\n"
    )
    with open(os.path.join(commit_dir, "network.py"), "w", encoding="utf-8") as f:
        f.write(net)
    for label in range(3):
        with open(os.path.join(commit_dir, "conf", f"Model_{label}.json"), "w", encoding="utf-8") as f:
            json.dump({"model": "GRU", "hidden": CFG["hidden"], "window": CFG["window"]}, f)
        torch.save(sds[label], os.path.join(commit_dir, "model", f"Model_{label}.pth"))


def run_task(name, train_dates, val_dates, device):
    t0 = time.time()
    Xtr, ytr, tid = load_block(train_dates)
    ckpt = os.path.join(OUT, f"ckpt_{name}")
    os.makedirs(ckpt, exist_ok=True)
    sds, losses = [], []
    for label in range(3):
        l = train_one(Xtr, ytr, label, tid, device, ckpt)
        sds.append(torch.load(os.path.join(ckpt, f"Model_{label}.pth"), map_location="cpu"))
        losses.append(l)
    commit_dir = os.path.join(OUT, f"commit_gru_{name}")
    build_commit(sds, commit_dir)
    ev = Evaluator(DATA, commit_dir, "cuda:0")
    pooled = ev.evaluate(date=val_dates)
    daily = {d: ev.evaluate(date=d) for d in val_dates}
    res = {"name": name, "train": train_dates, "val": val_dates,
           "pooled": pooled, "daily": daily, "losses": losses,
           "elapsed_sec": round(time.time() - t0, 1),
           "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1)}
    torch.cuda.reset_peak_memory_stats(device)
    return res


if __name__ == "__main__":
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    tasks = {
        "foldA": (list(range(0, 8)), [8, 9, 10, 11]),
        "foldB": (list(range(0, 12)), [12, 13, 14, 15]),
        "gap1": (list(range(0, 7)), [8, 9, 10, 11]),
    }
    results = {}
    for name, (tr, va) in tasks.items():
        print("running", name, flush=True)
        results[name] = run_task(name, tr, va, device)
        print(json.dumps({k: results[name][k] for k in ("pooled", "losses", "elapsed_sec", "peak_vram_mb")}, indent=2), flush=True)
    with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("done", flush=True)
