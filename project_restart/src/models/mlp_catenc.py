# -*- coding: utf-8 -*-
"""EXP-0009: category encoding ablation on strong-MLP. Single variable: f083-f099 as one-hot
(17x3=51 dims) replacing raw float, input 100 -> 134. Same strong-MLP recipe otherwise.
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
OUT = os.path.join("project_restart", "experiments", "EXP-0009")
os.makedirs(OUT, exist_ok=True)

CFG = {"hidden": 128, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 40, "batch_size": 8192,
       "seed": 42, "eval_every": 2, "patience": 6, "in_dim": 134}


def load_block(dates):
    lf = pl.scan_parquet(DATA).filter(pl.col("date").is_in(sorted(dates)))
    frame = lf.sort(["date", "time", "code"]).collect()
    feats = sorted([n for n in frame.columns if n.startswith("f")], key=lambda n: int(n[1:]))
    X = frame.select(feats).to_numpy().astype(np.float32)
    y = frame.select(["y0", "y1", "y2"]).to_numpy().astype(np.float32)
    return X, y


def one_hot_cats(X):
    cont = X[:, :83].astype(np.float32)
    cats = X[:, 83:].astype(np.int64)
    oh = np.eye(3, dtype=np.float32)[cats].reshape(len(X), -1)
    return np.concatenate([cont, oh], axis=1)


class MLP(nn.Module):
    def __init__(self, hidden=128, in_dim=134):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.network(x)


def predict_batched(model, X, device, chunk=1 << 18):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            outs.append(model(torch.from_numpy(X[s:s + chunk]).to(device)).cpu().numpy())
    return np.concatenate(outs)


def train_one(Xtr, ytr, label, Xva, yva, device, ckpt_dir, seed):
    torch.manual_seed(seed)
    model = MLP(CFG["hidden"], CFG["in_dim"]).to(device)
    X = torch.from_numpy(Xtr).to(device)
    Y = torch.from_numpy(ytr[:, label:label + 1].astype(np.float32)).to(device)
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
            loss = nn.functional.mse_loss(model(X[I]), Y[I])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.detach().item() * len(I)
            cnt += len(I)
        train_loss = total / cnt
        if (epoch + 1) % CFG["eval_every"] == 0:
            p = predict_batched(model, Xva, device)
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
        "class MLP(BaseModel):\n"
        "    def __init__(self, config):\n"
        "        super().__init__(config)\n"
        "        hidden = config['hidden']\n"
        "        self.network = nn.Sequential(nn.Linear(134, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))\n"
        "    def forward(self, x):\n"
        "        return self.network(x)\n"
        "    def reset(self):\n"
        "        pass\n"
        "    def predict(self, x):\n"
        "        cont = x[:, :83]\n"
        "        cats = x[:, 83:].long()\n"
        "        oh = F.one_hot(cats, num_classes=3).float().reshape(x.shape[0], -1)\n"
        "        return self.forward(torch.cat([cont, oh], dim=1))\n"
    )
    with open(os.path.join(commit_dir, "network.py"), "w", encoding="utf-8") as f:
        f.write(net)
    for label in range(3):
        with open(os.path.join(commit_dir, "conf", f"Model_{label}.json"), "w", encoding="utf-8") as f:
            json.dump({"model": "MLP", "hidden": CFG["hidden"]}, f)
        torch.save(sds[label], os.path.join(commit_dir, "model", f"Model_{label}.pth"))


def run_task(name, train_dates, val_dates, device):
    t0 = time.time()
    Xtr, ytr = load_block(train_dates)
    Xva, yva = load_block(val_dates)
    Xtr = one_hot_cats(Xtr)
    Xva = one_hot_cats(Xva)
    ckpt = os.path.join(OUT, f"ckpt_{name}")
    os.makedirs(ckpt, exist_ok=True)
    sds, bests, hists = [], [], []
    for label in range(3):
        b, h = train_one(Xtr, ytr, label, Xva, yva, device, ckpt, CFG["seed"])
        sds.append(torch.load(os.path.join(ckpt, f"Model_{label}.pth"), map_location="cpu"))
        bests.append({k: v for k, v in b.items() if k != "sd"})
        hists.append(h)
    commit_dir = os.path.join(OUT, f"commit_cat_{name}")
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
