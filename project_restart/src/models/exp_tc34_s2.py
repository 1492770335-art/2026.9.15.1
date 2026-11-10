# -*- coding: utf-8 -*-
"""TC-34 Gate S2 F1: train the frozen 9-submodel routing on date0-19 (full 20 days), fixed E.
MarketGRU K16 (300-dim MKT-v1) y0 E=4 / y1 E=2, seeds 42/43/44 (6 models);
non-Market K16 GRU100 y2 E=2, seeds 42/43/44 (3 models).
No epoch search on full20. Phased loading (market data then k16 data) to avoid RAM OOM.
"""
import gc
import hashlib
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
from gru_strong import load_block, make_batch, CFG  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0023")
os.makedirs(OUT, exist_ok=True)

K = 16
CFG["window"] = K
ALL_DATES = list(range(20))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


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


def train_full(seed, label, is_market, E, X, y, tid, device):
    torch.manual_seed(seed)
    model = (GRU300(CFG["hidden"]) if is_market else GRU100(CFG["hidden"])).to(device)
    Y = torch.from_numpy(y[:, label:label + 1].astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    n = len(X)
    bpe = math.ceil(n / CFG["batch_size"])
    losses = []
    t0 = time.time()
    for epoch in range(E):
        rng = np.random.default_rng(seed + epoch)
        order = rng.permutation(n)
        losses.append(train_epoch_fn(model, X, Y, tid, order, opt, device))
    tag = ("market" if is_market else "k16") + f"_full_s{seed}_y{label}"
    out_path = os.path.join(OUT, tag + ".pth")
    torch.save(model.state_dict(), out_path)
    return {"seed": seed, "label": label, "is_market": is_market, "E": E, "steps": bpe * E,
            "losses": losses, "elapsed_sec": round(time.time() - t0, 1), "sha256": sha256(out_path),
            "ckpt": out_path}


def load_full(is_market):
    X, y, tid = load_block(ALL_DATES)
    if is_market:
        X = add_market(X)
    return X, y, tid


if __name__ == "__main__":
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device, "K:", K, "E: y0=4 y1=2 y2=2", flush=True)
    results = {"train": [], "start": time.strftime("%Y-%m-%dT%H:%M:%S")}
    t_all = time.time()
    # market models (6)
    Xm, ym, tdm = load_full(True)
    for seed in (42, 43, 44):
        for label, E in ((0, 4), (1, 2)):
            r = train_full(seed, label, True, E, Xm, ym, tdm, device)
            results["train"].append(r)
            print(json.dumps(r), flush=True)
    del Xm, ym, tdm
    gc.collect()
    torch.cuda.empty_cache()
    # k16 y2 models (3)
    Xk, yk, tdk = load_full(False)
    for seed in (42, 43, 44):
        r = train_full(seed, 2, False, 2, Xk, yk, tdk, device)
        results["train"].append(r)
        print(json.dumps(r), flush=True)
    del Xk, yk, tdk
    gc.collect()
    torch.cuda.empty_cache()
    results["total_elapsed_sec"] = round(time.time() - t_all, 1)
    results["end"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    results["peak_vram_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
    results["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved(device) / 1e6, 1)
    with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print(json.dumps({k: results[k] for k in ("total_elapsed_sec", "peak_vram_allocated_mb", "peak_vram_reserved_mb")}, indent=2), flush=True)
    print("done", flush=True)
