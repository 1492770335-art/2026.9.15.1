# -*- coding: utf-8 -*-
"""EXP-0010: Pearson-mixed loss for y2 on strong-MLP host.
Variable: alpha in {0, 0.1, 0.3, 1.0}; loss = MSE + alpha * (1 - batch_pearson).
Only y2 uses the mixed loss; y0/y1 use MSE (trained once per fold, shared across alpha runs).
Fold A grid -> fold B confirm. Official Evaluator for final numbers.
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
OUT = os.path.join("project_restart", "experiments", "EXP-0010")
os.makedirs(OUT, exist_ok=True)

CFG = {"hidden": 128, "lr": 1e-3, "weight_decay": 1e-4, "epochs": 40, "batch_size": 8192,
       "seed": 42, "eval_every": 2, "patience": 6, "alphas": [0.0, 0.1, 0.3, 1.0]}


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


def batch_pearson_loss(p, y, eps=1e-10):
    p = p - p.mean()
    y = y - y.mean()
    cov = (p * y).sum()
    denom = ((p * p).sum() * (y * y).sum()).clamp_min(eps).sqrt()
    return 1.0 - cov / denom


def predict_batched(model, X, device, chunk=1 << 18):
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), chunk):
            outs.append(model(torch.from_numpy(X[s:s + chunk]).to(device)).cpu().numpy())
    return np.concatenate(outs)


def train_one(Xtr, ytr, label, Xva, yva, device, ckpt_dir, seed, alpha=0.0):
    torch.manual_seed(seed)
    model = MLP(CFG["hidden"]).to(device)
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
            p = model(X[I])
            mse = nn.functional.mse_loss(p, Y[I])
            loss = mse + alpha * batch_pearson_loss(p.squeeze(1), Y[I].squeeze(1))
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


def run_fold(name, train_dates, val_dates, device, alphas):
    t0 = time.time()
    Xtr, ytr = load_block(train_dates)
    Xva, yva = load_block(val_dates)
    ckpt = os.path.join(OUT, f"ckpt_{name}")
    os.makedirs(ckpt, exist_ok=True)
    # y0/y1 once with MSE
    sd01 = []
    for label in (0, 1):
        b, _ = train_one(Xtr, ytr, label, Xva, yva, device, ckpt, CFG["seed"], alpha=0.0)
        sd01.append(torch.load(os.path.join(ckpt, f"Model_{label}.pth"), map_location="cpu"))
    out = {}
    for alpha in alphas:
        b, h = train_one(Xtr, ytr, 2, Xva, yva, device, ckpt, CFG["seed"], alpha=alpha)
        sd2 = torch.load(os.path.join(ckpt, "Model_2.pth"), map_location="cpu")
        sds = [sd01[0], sd01[1], sd2]
        commit_dir = os.path.join(OUT, f"commit_{name}_a{alpha}")
        build_commit(sds, commit_dir)
        ev = Evaluator(DATA, commit_dir, "cuda:0")
        pooled = ev.evaluate(date=val_dates)
        daily = {d: ev.evaluate(date=d) for d in val_dates}
        out[str(alpha)] = {"pooled": pooled, "daily": daily, "y2_best": {k: v for k, v in b.items() if k != "sd"},
                           "y2_history": h}
        print(json.dumps({"fold": name, "alpha": alpha, "pooled": pooled}, indent=2), flush=True)
    out["elapsed_sec"] = round(time.time() - t0, 1)
    return out


if __name__ == "__main__":
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = {}
    results["foldA"] = run_fold("foldA", list(range(0, 8)), [8, 9, 10, 11], device, CFG["alphas"])
    results["foldB"] = run_fold("foldB", list(range(0, 12)), [12, 13, 14, 15], device, CFG["alphas"])
    with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("done", flush=True)
