# -*- coding: utf-8 -*-
"""TC-32 Gate H1: train Market y0/y1 for seeds 43/44 x folds A/B (8 configs), then build
3-seed routing ensemble (y0/y1 = Market seeds 42/43/44 equal-weight; y2 = K16 non-Market
seeds 42/43/44 equal-weight) and evaluate via OFFICIAL Evaluator. Same K16-GRU + MKT-v1 (300-dim),
new inner early-stop protocol, train/eval bugfix. seed42 market = EXP-0019; seed42/43/44 K16 = EXP-0016/0017/0018.
"""
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
OUT = os.path.join("project_restart", "experiments", "EXP-0021")
os.makedirs(OUT, exist_ok=True)

K = 16
CFG["window"] = K
IN_DIM = 300
CAP = 30
PATIENCE = 4
EVAL_EVERY = 2

FOLDS = {
    "A": {"inner_train": [0, 1, 2, 3, 4, 5], "inner_stop": [6, 7], "full_train": list(range(0, 8)), "outer_val": [8, 9, 10, 11]},
    "B": {"inner_train": list(range(0, 10)), "inner_stop": [10, 11], "full_train": list(range(0, 12)), "outer_val": [12, 13, 14, 15]},
}
# K16 non-market per-seed per-target corr (for paired comparison): {fold: {seed: {target: corr}}}
K16_CORR = {
    "A": {42: {"y0": 0.4501, "y1": 0.2874}, 43: {"y0": 0.4492, "y1": 0.2923}, 44: {"y0": 0.4510, "y1": 0.2954}},
    "B": {42: {"y0": 0.4439, "y1": 0.2905}, 43: {"y0": 0.4431, "y1": 0.2918}, 44: {"y0": 0.4463, "y1": 0.2940}},
}
K16_Y2_3SEED_CORR = {"A": 0.1374, "B": 0.1791}


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


def select_epoch(X, y, tid, label, Xst, yst, tid_st, device, seed):
    torch.manual_seed(seed)
    model = GRU300(CFG["hidden"]).to(device)
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


def retrain_full(X, y, tid, label, E, device, out_path, seed):
    torch.manual_seed(seed)
    model = GRU300(CFG["hidden"]).to(device)
    Y = torch.from_numpy(y[:, label:label + 1].astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    n = len(X)
    bpe = math.ceil(n / CFG["batch_size"])
    for epoch in range(E):
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(n)
        train_epoch_fn(model, X, Y, tid, order, opt, device)
    torch.save(model.state_dict(), out_path)
    return bpe * E


def train_market(seed, fold, label, device):
    spec = FOLDS[fold]
    Xi, yi, tii = load_block(spec["inner_train"])
    Xs, ys, tis = load_block(spec["inner_stop"])
    Xf, yf, tif = load_block(spec["full_train"])
    Xi, Xs, Xf = add_market(Xi), add_market(Xs), add_market(Xf)
    E = select_epoch(Xi, yi, tii, label, Xs, ys, tis, device, seed)
    out_path = os.path.join(OUT, f"market_s{seed}_{fold}_y{label}.pth")
    steps = retrain_full(Xf, yf, tif, label, E, device, out_path, seed)
    # single-target val corr (batched)
    Xv, yv, tiv = load_block(spec["outer_val"])
    Xv = add_market(Xv)
    m = GRU300(CFG["hidden"]).to(device)
    m.load_state_dict(torch.load(out_path, map_location=device, weights_only=True))
    m.eval()
    p = predict_batched(m, Xv, tiv, device)
    corr = float(pooled_pearson_raw(p, yv[:, label:label + 1])[0])
    return {"seed": seed, "fold": fold, "label": label, "E": E, "steps": steps, "corr": corr,
            "ckpt": out_path}


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


def build_routing_commit(fold):
    commit = os.path.join(OUT, f"commit_3seed_routing_{fold}")
    if os.path.exists(commit):
        shutil.rmtree(commit)
    os.makedirs(os.path.join(commit, "conf"))
    os.makedirs(os.path.join(commit, "model"))
    shutil.copy(os.path.join("project", "project", "script", "base.py"), os.path.join(commit, "base.py"))
    with open(os.path.join(commit, "network.py"), "w", encoding="utf-8") as f:
        f.write(NET)
    # market y0/y1 members: seed42 from EXP-0019, seed43/44 from this run
    market_src = os.path.join("project_restart", "experiments", "EXP-0019", f"commit_mkt_{fold}", "model")
    for label, cls in ((0, "MarketEnsemble"), (1, "MarketEnsemble"), (2, "K16Ensemble")):
        ens_sd = {}
        if cls == "MarketEnsemble":
            for i, seed in enumerate((42, 43, 44)):
                p = (os.path.join(market_src, f"Model_{label}.pth") if seed == 42
                     else os.path.join(OUT, f"market_s{seed}_{fold}_y{label}.pth"))
                state = torch.load(p, map_location="cpu", weights_only=True)
                for k, v in state.items():
                    ens_sd[f"members.{i}.{k}"] = v
        else:  # K16Ensemble y2: seeds 42/43/44 K16 non-market
            k16_src = {
                42: os.path.join("project_restart", "experiments", "EXP-0016", f"commit_K16_{fold}", "model", "Model_2.pth"),
                43: os.path.join("project_restart", "experiments", "EXP-0017", f"commit_s43_K16_{fold}", "model", "Model_2.pth"),
                44: os.path.join("project_restart", "experiments", "EXP-0017", f"commit_s44_K16_{fold}", "model", "Model_2.pth"),
            }
            for i, seed in enumerate((42, 43, 44)):
                state = torch.load(k16_src[seed], map_location="cpu", weights_only=True)
                for k, v in state.items():
                    ens_sd[f"members.{i}.{k}"] = v
        torch.save(ens_sd, os.path.join(commit, "model", f"Model_{label}.pth"))
        with open(os.path.join(commit, "conf", f"Model_{label}.json"), "w", encoding="utf-8") as f:
            json.dump({"model": cls, "hidden": 128, "window": 16}, f)
    return commit


if __name__ == "__main__":
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device, flush=True)
    results = {"market_train": [], "routing": {}}
    for seed in (43, 44):
        for fold in ("A", "B"):
            for label in (0, 1):
                r = train_market(seed, fold, label, device)
                results["market_train"].append(r)
                print(json.dumps(r), flush=True)
    # build 3-seed routing and evaluate
    for fold in ("A", "B"):
        commit = build_routing_commit(fold)
        ev = Evaluator(DATA, commit, device)
        pooled = ev.evaluate(date=FOLDS[fold]["outer_val"])
        daily = {int(d): ev.evaluate(date=d) for d in FOLDS[fold]["outer_val"]}
        extra = {}
        if fold == "A":
            extra["no_date8"] = ev.evaluate(date=[9, 10, 11])
        t0 = time.time()
        ev.evaluate(date=FOLDS[fold]["outer_val"][0])
        lat = time.time() - t0
        results["routing"][fold] = {"pooled_official": pooled, "daily": daily, "extra": extra,
                                    "latency_sec_day": round(lat, 1),
                                    "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1)}
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        print(f"[routing {fold}] pooled={pooled} lat={lat:.1f}s", flush=True)
    with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("done", flush=True)
