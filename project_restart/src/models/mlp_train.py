# -*- coding: utf-8 -*-
"""EXP-0002: official-as-is SiLU MLP 100->128->128->1 baselines on A/B folds + gap1 sensitivity.
Recipe mirrors example/fit.py exactly (AdamW lr=1e-4 wd=1e-4, MSE, grad-clip 1.0, 2 epochs,
batch 8192, shuffle=False, seed 42), but with strict date splits. Verified via OFFICIAL Evaluator (cuda).
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

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0002")
os.makedirs(OUT, exist_ok=True)

CFG = {"hidden": 128, "lr": 1e-4, "weight_decay": 1e-4, "epochs": 2, "batch_size": 8192, "seed": 42}


def load_block(dates):
    lf = pl.scan_parquet(DATA).filter(pl.col("date").is_in(sorted(dates)))
    frame = lf.sort(["date", "time", "code"]).collect()
    feats = sorted([n for n in frame.columns if n.startswith("f")], key=lambda n: int(n[1:]))
    X = frame.select(feats).to_numpy().astype(np.float32)
    y = frame.select(["y0", "y1", "y2"]).to_numpy().astype(np.float32)
    return X, y


class MLP(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(100, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.network(x)


def train_one(Xtr, ytr, label, device, ckpt_dir):
    torch.manual_seed(CFG["seed"])
    model = MLP(CFG["hidden"]).to(device)
    X = torch.from_numpy(Xtr).to(device)
    Y = torch.from_numpy(ytr[:, label:label + 1].astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    losses = []
    n = X.shape[0]
    for epoch in range(CFG["epochs"]):
        total, cnt = 0.0, 0
        for s in range(0, n, CFG["batch_size"]):
            xb = X[s:s + CFG["batch_size"]]
            yb = Y[s:s + CFG["batch_size"]]
            loss = nn.functional.mse_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.detach().item() * xb.shape[0]
            cnt += xb.shape[0]
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
        "from torch import nn\n"
        "from .base import BaseModel\n"
        "class MLP(BaseModel):\n"
        "    def __init__(self, config):\n"
        "        super().__init__(config)\n"
        "        hidden = config['hidden']\n"
        "        self.network = nn.Sequential(nn.Linear(100, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))\n"
        "    def forward(self, x):\n"
        "        return self.network(x)\n"
        "    def reset(self):\n"
        "        pass\n"
        "    def predict(self, x):\n"
        "        return self.forward(x)\n"
    )
    with open(os.path.join(commit_dir, "network.py"), "w", encoding="utf-8") as f:
        f.write(net)
    for label in range(3):
        with open(os.path.join(commit_dir, "conf", f"Model_{label}.json"), "w", encoding="utf-8") as f:
            json.dump({"model": "MLP", "hidden": CFG["hidden"]}, f)
        torch.save(sds[label], os.path.join(commit_dir, "model", f"Model_{label}.pth"))


def run_task(name, train_dates, val_dates, device):
    t0 = time.time()
    torch.manual_seed(CFG["seed"])
    Xtr, ytr = load_block(train_dates)
    ckpt = os.path.join(OUT, f"ckpt_{name}")
    os.makedirs(ckpt, exist_ok=True)
    sds, losses = [], []
    for label in range(3):
        sd = None
        # train with save interleaved: train_one saves the pth; we then reload it
        l = train_one(Xtr, ytr, label, device, ckpt)
        sd = torch.load(os.path.join(ckpt, f"Model_{label}.pth"), map_location="cpu")
        sds.append(sd)
        losses.append(l)
    commit_dir = os.path.join(OUT, f"commit_mlp_{name}")
    build_commit(sds, commit_dir)
    ev = Evaluator(DATA, commit_dir, "cuda:0")
    pooled = ev.evaluate(date=val_dates)
    daily = {d: ev.evaluate(date=d) for d in val_dates}
    # train-internal diagnostic only
    tr_ev = ev.evaluate(date=train_dates)
    res = {"name": name, "train": train_dates, "val": val_dates,
           "pooled": pooled, "daily": daily, "train_internal_diagnostic": tr_ev,
           "losses": losses, "elapsed_sec": round(time.time() - t0, 1),
           "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1)}
    torch.cuda.reset_peak_memory_stats(device)
    return res


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    tasks = {
        "foldA": (list(range(0, 8)), [8, 9, 10, 11]),
        "foldB": (list(range(0, 12)), [12, 13, 14, 15]),
        "gap1": (list(range(0, 7)), [8, 9, 10, 11]),
    }
    results = {}
    for name, (tr, va) in tasks.items():
        print("running", name)
        results[name] = run_task(name, tr, va, device)
        print(json.dumps({k: results[name][k] for k in ("pooled", "losses", "elapsed_sec", "peak_vram_mb")}, indent=2))
    with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("done")
