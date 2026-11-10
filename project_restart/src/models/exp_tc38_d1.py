# -*- coding: utf-8 -*-
"""TC-38 Gate D1: y2-only K16-GRU + diff83 (183-dim) seed42 x Fold A/B.
2 target-fold configs, each = inner-select E (phase1) + full fixed-E retrain (phase2) = 4 phases.
Control = EXP-0016 K16-GRU seed42 y2 (raw c2 A 0.125503 / B 0.170568). Batched + streaming consistency.
"""
import gc
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import polars as pl
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
sys.path.insert(0, os.path.join("project_restart", "src", "models"))
sys.path.insert(0, os.path.join("project_restart", "src", "features"))
from gru_strong import make_batch, CFG  # noqa: E402
from src.features.diff83 import build_diff83  # noqa: E402
from src.eval.metrics import pooled_pearson_raw  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0025")
os.makedirs(OUT, exist_ok=True)

K = 16
CFG["window"] = K
CAP = 30
PATIENCE = 4
EVAL_EVERY = 2
SEED = 42
LABEL = 2

FOLDS = {
    "A": {"inner_train": [0, 1, 2, 3, 4, 5], "inner_stop": [6, 7], "full_train": list(range(0, 8)), "outer_val": [8, 9, 10, 11]},
    "B": {"inner_train": list(range(0, 10)), "inner_stop": [10, 11], "full_train": list(range(0, 12)), "outer_val": [12, 13, 14, 15]},
}
CONTROL_RAW_C2 = {"A": 0.125503, "B": 0.170568}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_block_dates(dates):
    lf = pl.scan_parquet(DATA).filter(pl.col("date").is_in(sorted(dates)))
    frame = lf.sort(["date", "time", "code"]).collect()
    feats = sorted([n for n in frame.columns if n.startswith("f")], key=lambda n: int(n[1:]))
    X = frame.select(feats).to_numpy().astype(np.float32)
    y = frame.select(["y0", "y1", "y2"]).to_numpy().astype(np.float32)
    d = frame["date"].to_numpy()
    changed = np.r_[True, d[1:] != d[:-1]]
    day_start_row = np.maximum.accumulate(np.where(changed, np.arange(len(d)), 0))
    tick_in_day = ((np.arange(len(d)) - day_start_row) // 50).astype(np.int64)
    return X, y, d, tick_in_day


class GRU183(nn.Module):
    def __init__(self, hidden=128):
        super().__init__()
        self.gru = nn.GRU(183, hidden, batch_first=True)
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


def select_epoch(X, y, tid, Xst, yst, tid_st, device):
    torch.manual_seed(SEED)
    model = GRU183(CFG["hidden"]).to(device)
    Y = torch.from_numpy(y[:, LABEL:LABEL + 1].astype(np.float32)).to(device)
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
            c = pooled_pearson_raw(p, yst[:, LABEL:LABEL + 1])[0]
            traj.append({"epoch": epoch + 1, "train_loss": train_loss, "inner_stop_corr": float(c)})
            if c > best["c"]:
                best = {"epoch": epoch + 1, "c": float(c)}
            if epoch + 1 - best["epoch"] >= PATIENCE:
                break
    return best["epoch"], traj


def retrain_full(X, y, tid, E, device):
    torch.manual_seed(SEED)
    model = GRU183(CFG["hidden"]).to(device)
    Y = torch.from_numpy(y[:, LABEL:LABEL + 1].astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    n = len(X)
    bpe = math.ceil(n / CFG["batch_size"])
    losses = []
    for epoch in range(E):
        rng = np.random.default_rng(SEED + epoch)
        order = rng.permutation(n)
        losses.append(train_epoch_fn(model, X, Y, tid, order, opt, device))
    return model, bpe * E, losses


def daily_raw_c2(model, Xv, yv, dv, tiv, device):
    out = {}
    for d in sorted(set(dv.tolist())):
        m = dv == d
        p = predict_batched(model, Xv[m], tiv[m], device)
        out[int(d)] = float(pooled_pearson_raw(p, yv[m][:, LABEL:LABEL + 1])[0])
    return out


def streaming_vs_batch(model, Xv, yv, dv, tiv, device, first_day):
    """Stream the online GRU183 over one day (raw 100-dim ticks + prev_raw + history) and compare to batched."""
    rawX, _, rawD, _ = load_block_dates([first_day])
    Xr = rawX  # (N,100)
    model.eval()
    hist = None
    prev_raw = None
    stream = []
    with torch.no_grad():
        T = Xr.reshape(-1, 50, 100)
        for t in range(T.shape[0]):
            xt = torch.from_numpy(T[t]).to(device)  # (50,100)
            if prev_raw is None:
                dd = torch.zeros(50, 83, device=device, dtype=torch.float32)
            else:
                dd = xt[:, :83] - prev_raw[:, :83]
            prev_raw = xt
            xm = torch.cat([xt, dd], dim=1)  # (50,183)
            cur = xm[:, None]  # (50,1,183)
            hist = cur if hist is None else torch.cat([hist, cur], dim=1)
            hist = hist[:, -K:]  # (50,T,183)
            h = F.pad(hist, (0, 0, K - hist.shape[1], 0))  # (50,K,183)
            stream.append(model(h).cpu().numpy())  # (50,1)
    stream = np.concatenate(stream, axis=0)  # (N,1)
    m0 = dv == first_day
    Xv183 = build_diff83(Xv[m0], dv[m0])
    batch = predict_batched(model, Xv183, tiv[m0], device)
    max_abs = float(np.abs(stream - batch).max())
    return max_abs


def train_fold(fold, device):
    spec = FOLDS[fold]
    t0 = time.time()
    # phase 1: inner select E
    Xi, yi, di, tii = load_block_dates(spec["inner_train"])
    Xi = build_diff83(Xi, di)
    Xs, ys, ds, tis = load_block_dates(spec["inner_stop"])
    Xs = build_diff83(Xs, ds)
    E, traj = select_epoch(Xi, yi, tii, Xs, ys, tis, device)
    inner_bpe = math.ceil(len(Xi) / CFG["batch_size"])
    del Xi, Xs, yi, ys, di, ds, tii, tis
    gc.collect()
    torch.cuda.empty_cache()
    # phase 2: full retrain fixed E
    Xf, yf, df, tif = load_block_dates(spec["full_train"])
    Xf = build_diff83(Xf, df)
    model, steps, losses = retrain_full(Xf, yf, tif, E, device)
    del Xf, yf, df, tif
    gc.collect()
    torch.cuda.empty_cache()
    ckpt = os.path.join(OUT, f"d1_s{SEED}_{fold}_y2.pth")
    torch.save(model.state_dict(), ckpt)
    # eval on outer (batched raw c2 + daily)
    Xv, yv, dv, tiv = load_block_dates(spec["outer_val"])
    Xv183 = build_diff83(Xv, dv)
    m = GRU183(CFG["hidden"]).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval()
    p_all = predict_batched(m, Xv183, tiv, device)
    raw_c2 = float(pooled_pearson_raw(p_all, yv[:, LABEL:LABEL + 1])[0])
    daily = daily_raw_c2(m, Xv183, yv, dv, tiv, device)
    # streaming vs batch consistency on first outer day
    max_abs = streaming_vs_batch(m, Xv, yv, dv, tiv, device, spec["outer_val"][0])
    rec = {"fold": fold, "E": E, "inner_traj": traj, "inner_bpe": inner_bpe,
           "full_steps": steps, "full_losses": losses, "raw_c2": raw_c2, "daily_raw_c2": daily,
           "control_raw_c2": CONTROL_RAW_C2[fold], "delta_c2": raw_c2 - CONTROL_RAW_C2[fold],
           "delta_c2_weighted": 0.5 * (raw_c2 - CONTROL_RAW_C2[fold]),
           "streaming_batch_max_abs": max_abs, "ckpt_sha256": sha256(ckpt),
           "elapsed_sec": round(time.time() - t0, 1)}
    torch.cuda.reset_peak_memory_stats(device)
    return rec


if __name__ == "__main__":
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device, "LABEL:", LABEL, "input_dim: 183", flush=True)
    results = {"folds": {}}
    for fold in ("A", "B"):
        r = train_fold(fold, device)
        results["folds"][fold] = r
        with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(json.dumps({k: r[k] for k in ("fold", "E", "raw_c2", "daily_raw_c2", "delta_c2", "delta_c2_weighted", "streaming_batch_max_abs", "full_steps")}, indent=2, default=str), flush=True)
    results["peak_vram_allocated_mb"] = round(torch.cuda.max_memory_allocated(device) / 1e6, 1)
    results["peak_vram_reserved_mb"] = round(torch.cuda.max_memory_reserved(device) / 1e6, 1)
    with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("done", flush=True)
