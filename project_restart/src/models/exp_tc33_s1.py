# -*- coding: utf-8 -*-
"""TC-33 Gate S1: train 9 networks with frozen recipe on S1 split, then evaluate date16-19 once.
Split: inner_fit 0-11, inner_stop 12-15, full_fit 0-15, outer_val 16-19.
Models: MarketGRU K16 (300-dim MKT-v1) y0/y1 x seeds 42/43/44 (6); non-Market K16 GRU100 y2 x seeds 42/43/44 (3).
date16-19 used only for final official evaluation (not epoch selection).
"""
import gc
import json
import math
import os
import shutil
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
from src.eval.metrics import pooled_pearson_raw  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0022")
os.makedirs(OUT, exist_ok=True)

K = 16
CFG["window"] = K
CAP = 30
PATIENCE = 4
EVAL_EVERY = 2

SPLIT = {"inner_train": list(range(0, 12)), "inner_stop": [12, 13, 14, 15],
         "full_train": list(range(0, 16)), "outer_val": [16, 17, 18, 19]}


def add_market(X):
    T = X.reshape(-1, 50, 100)
    m = T.mean(axis=1, keepdims=True)
    s = T.std(axis=1, keepdims=True, ddof=1)
    m = np.broadcast_to(m, T.shape).reshape(-1, 100)
    s = np.broadcast_to(s, T.shape).reshape(-1, 100)
    return np.concatenate([X, m, s], axis=1).astype(np.float32)


class GRU300(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.gru = nn.GRU(300, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)

    def forward(self, x):
        h, _ = self.gru(x)
        return self.output(h[:, -1])


class GRU100(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.gru = nn.GRU(100, hidden, batch_first=True)
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


def select_epoch(model, X, y, tid, label, Xst, yst, tid_st, device, seed):
    torch.manual_seed(seed)
    Y = torch.from_numpy(y[:, label:label + 1].astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    n = len(X)
    best = {"epoch": -1, "c": -2.0}
    for epoch in range(CAP):
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(n)
        train_epoch_fn(model, X, Y, tid, order, opt, device)
        if (epoch + 1) % EVAL_EVERY == 0:
            p = predict_batched(model, Xst, tid_st, device)
            c = pooled_pearson_raw(p, yst[:, label:label + 1])[0]
            if c > best["c"]:
                best = {"epoch": epoch + 1, "c": float(c)}
            if epoch + 1 - best["epoch"] >= PATIENCE:
                break
    return best["epoch"]


def retrain_full(model, X, y, tid, label, E, device, seed):
    torch.manual_seed(seed)
    Y = torch.from_numpy(y[:, label:label + 1].astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    n = len(X)
    bpe = math.ceil(n / CFG["batch_size"])
    for epoch in range(E):
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(n)
        train_epoch_fn(model, X, Y, tid, order, opt, device)
    return bpe * E


def train_one(seed, label, is_market, device):
    # phase 1: select E on inner (only inner data in memory)
    Xi, yi, tii = load_block(SPLIT["inner_train"])
    Xs, ys, tis = load_block(SPLIT["inner_stop"])
    if is_market:
        Xi = add_market(Xi)
        Xs = add_market(Xs)
    model = (GRU300(CFG["hidden"]) if is_market else GRU100(CFG["hidden"])).to(device)
    E = select_epoch(model, Xi, yi, tii, label, Xs, ys, tis, device, seed)
    del Xi, Xs, yi, ys, tii, tis
    gc.collect()
    torch.cuda.empty_cache()
    # phase 2: retrain on full (only full data in memory)
    Xf, yf, tif = load_block(SPLIT["full_train"])
    if is_market:
        Xf = add_market(Xf)
    model2 = (GRU300(CFG["hidden"]) if is_market else GRU100(CFG["hidden"])).to(device)
    steps = retrain_full(model2, Xf, yf, tif, label, E, device, seed)
    del Xf, yf, tif
    gc.collect()
    torch.cuda.empty_cache()
    tag = ("market" if is_market else "k16") + f"_s1_s{seed}_y{label}"
    out_path = os.path.join(OUT, tag + ".pth")
    torch.save(model2.state_dict(), out_path)
    return {"seed": seed, "label": label, "is_market": is_market, "E": E, "steps": steps, "ckpt": out_path}


NET = (
    "import torch\n"
    "from torch import nn\n"
    "from torch.nn import functional as F\n"
    "from .base import BaseModel\n"
    "class _GRU100(nn.Module):\n"
    "    def __init__(self, hidden):\n"
    "        super().__init__()\n"
    "        self.gru = nn.GRU(100, hidden, batch_first=True)\n"
    "        self.output = nn.Linear(hidden, 1)\n"
    "    def forward(self, x):\n"
    "        h, _ = self.gru(x)\n"
    "        return self.output(h[:, -1])\n"
    "class _GRU300(nn.Module):\n"
    "    def __init__(self, hidden):\n"
    "        super().__init__()\n"
    "        self.gru = nn.GRU(300, hidden, batch_first=True)\n"
    "        self.output = nn.Linear(hidden, 1)\n"
    "    def forward(self, x):\n"
    "        h, _ = self.gru(x)\n"
    "        return self.output(h[:, -1])\n"
    "class MarketEnsemble(BaseModel):\n"
    "    def __init__(self, config):\n"
    "        super().__init__(config)\n"
    "        hidden = config['hidden']\n"
    "        self.window = config.get('window', 16)\n"
    "        self.members = nn.ModuleList([_GRU300(hidden) for _ in range(3)])\n"
    "        self.hist = [None, None, None]\n"
    "    def reset(self):\n"
    "        self.hist = [None, None, None]\n"
    "    def predict(self, x):\n"
    "        m = x.mean(0, keepdim=True)\n"
    "        s = x.std(0, keepdim=True)\n"
    "        xm = torch.cat([x, m.expand_as(x), s.expand_as(x)], dim=1)\n"
    "        cur = xm[:, None].detach()\n"
    "        outs = []\n"
    "        for k, mem in enumerate(self.members):\n"
    "            self.hist[k] = cur if self.hist[k] is None else torch.cat([self.hist[k], cur], dim=1)\n"
    "            self.hist[k] = self.hist[k][:, -self.window:]\n"
    "            h = F.pad(self.hist[k], (0, 0, self.window - self.hist[k].shape[1], 0))\n"
    "            outs.append(mem(h))\n"
    "        return torch.stack(outs, 0).mean(0)\n"
    "class K16Ensemble(BaseModel):\n"
    "    def __init__(self, config):\n"
    "        super().__init__(config)\n"
    "        hidden = config['hidden']\n"
    "        self.window = config.get('window', 16)\n"
    "        self.members = nn.ModuleList([_GRU100(hidden) for _ in range(3)])\n"
    "        self.hist = [None, None, None]\n"
    "    def reset(self):\n"
    "        self.hist = [None, None, None]\n"
    "    def predict(self, x):\n"
    "        cur = x[:, None].detach()\n"
    "        outs = []\n"
    "        for k, mem in enumerate(self.members):\n"
    "            self.hist[k] = cur if self.hist[k] is None else torch.cat([self.hist[k], cur], dim=1)\n"
    "            self.hist[k] = self.hist[k][:, -self.window:]\n"
    "            h = F.pad(self.hist[k], (0, 0, self.window - self.hist[k].shape[1], 0))\n"
    "            outs.append(mem(h))\n"
    "        return torch.stack(outs, 0).mean(0)\n"
)


def build_commit():
    commit = os.path.join(OUT, "commit_s1_routing")
    if os.path.exists(commit):
        shutil.rmtree(commit)
    os.makedirs(os.path.join(commit, "conf"))
    os.makedirs(os.path.join(commit, "model"))
    shutil.copy(os.path.join("project", "project", "script", "base.py"), os.path.join(commit, "base.py"))
    with open(os.path.join(commit, "network.py"), "w", encoding="utf-8") as f:
        f.write(NET)
    for label, cls, tagprefix in ((0, "MarketEnsemble", "market_s1"), (1, "MarketEnsemble", "market_s1"), (2, "K16Ensemble", "k16_s1")):
        ens_sd = {}
        for i, seed in enumerate((42, 43, 44)):
            p = os.path.join(OUT, f"{tagprefix}_s{seed}_y{label}.pth")
            state = torch.load(p, map_location="cpu", weights_only=True)
            for k, v in state.items():
                ens_sd[f"members.{i}.{k}"] = v
        torch.save(ens_sd, os.path.join(commit, "model", f"Model_{label}.pth"))
        with open(os.path.join(commit, "conf", f"Model_{label}.json"), "w", encoding="utf-8") as f:
            json.dump({"model": cls, "hidden": 128, "window": 16}, f)
    return commit


if __name__ == "__main__":
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device, "SPLIT:", SPLIT, flush=True)
    results = {"train": []}
    for seed in (42, 43, 44):
        for label in (0, 1):
            results["train"].append(train_one(seed, label, True, device))
            print(json.dumps(results["train"][-1]), flush=True)
    for seed in (42, 43, 44):
        results["train"].append(train_one(seed, 2, False, device))
        print(json.dumps(results["train"][-1]), flush=True)
    commit = build_commit()
    ev = Evaluator(DATA, commit, device)
    pooled = ev.evaluate(date=[16, 17, 18, 19])
    daily = {int(d): ev.evaluate(date=d) for d in [16, 17, 18, 19]}
    results["outer_16_19"] = {"pooled_official": pooled, "daily": daily}
    with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps(results["outer_16_19"], indent=2, default=str), flush=True)
    print("done", flush=True)
