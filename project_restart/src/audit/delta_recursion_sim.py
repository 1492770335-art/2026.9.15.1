# -*- coding: utf-8 -*-
"""Pure-delta recursion diagnostic (CPU, train-only OLS, no deep training).
A (legal): dhat = a + b*f000 fit on TRAIN 0-7; recursion yhat_0=dhat_0, yhat_t=yhat_{t-1}+dhat_t on VAL 8-11.
B (ILLEGAL_FOR_SUBMISSION, diagnostic only): same but anchor yhat_0 = TRUE y_0 (isolates accumulation
vs day-start anchor error; never usable in submission).
Compare pooled corr with frozen strong-GRU (0.4518/0.285/0.0938).
"""
import json
import os
import time

import numpy as np
import polars as pl

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "reports", "ph1_d03b_pure_delta_recursion.json")
t0 = time.time()
res = {}


def load(dates):
    lf = pl.scan_parquet(DATA).filter(pl.col("date").is_in(sorted(dates))).sort(["date", "time", "code"])
    f = lf.select(["date", "f000", "y0", "y1", "y2"]).collect()
    d = f["date"].to_numpy()
    f000 = f["f000"].to_numpy().astype(np.float64)
    y = f.select(["y0", "y1", "y2"]).to_numpy().astype(np.float64)
    return d, f000, y


def delta_of(y, dates):
    n = len(y)
    idx = np.arange(n)
    ok = (idx >= 50) & (dates == dates[np.clip(idx - 50, 0, n - 1)])
    out = y.copy()
    out[ok] = y[ok] - y[np.clip(idx[ok] - 50, 0, n - 1)]
    return out


def reconstruct(d, dates):
    L = d.reshape(-1, 50).astype(np.float64)
    T = L.shape[0]
    tick_day = dates.reshape(-1, 50)[:, 0]
    changed = np.r_[True, tick_day[1:] != tick_day[:-1]]
    dst = np.maximum.accumulate(np.where(changed, np.arange(T), 0))
    cs = np.cumsum(L, axis=0)
    off = np.zeros_like(L)
    v = dst > 0
    off[v] = cs[dst[v] - 1]
    return (cs - off).reshape(-1)


dtr, f000_tr, ytr = load(list(range(0, 8)))
dva, f000_va, yva = load([8, 9, 10, 11])

for j, yn in enumerate(["y0", "y1", "y2"]):
    dtrue_tr = delta_of(ytr[:, j], dtr)
    dtrue_va = delta_of(yva[:, j], dva)
    # train-only OLS: dtrue ~ [1, f000]
    A = np.stack([np.ones_like(f000_tr), f000_tr], 1)
    w, *_ = np.linalg.lstsq(A, dtrue_tr, rcond=None)
    dhat_va = w[0] + w[1] * f000_va
    c_in = float(np.corrcoef(dhat_va, dtrue_va)[0, 1])
    # A: legal recursion, day start yhat_0 = dhat_0
    levA = reconstruct(dhat_va, dva)
    # B: ILLEGAL anchor: yhat_0 = true y_0 at each day's first tick
    levB = levA.copy()
    first = np.r_[True, dva[1:] != dva[:-1]]
    levB[first] = yva[first, j]
    cA = float(np.corrcoef(levA, yva[:, j])[0, 1])
    cB = float(np.corrcoef(levB, yva[:, j])[0, 1])
    res[yn] = {
        "val_corr_dhat_with_dtrue": c_in,
        "ols_w": w.tolist(),
        "A_legal_pure_recursion_level_corr": cA,
        "B_illegal_true_anchor_level_corr": cB,
        "frozen_gru_level_corr": [0.4518, 0.285, 0.0938][j],
        "delta_vs_gru_A": cA - [0.4518, 0.285, 0.0938][j],
    }
    print(json.dumps({yn: res[yn]}, indent=2), flush=True)

res["note"] = ("A: 合法（仅训练折拟合的 OLS，无真实标签推理）；"
               "B: ILLEGAL_FOR_SUBMISSION（日初用真实 y_0 锚定，仅用于区分锚点误差与积累误差）；"
               "均非训练、非提交方案")
res["elapsed_sec"] = round(time.time() - t0, 1)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, default=str)
print("OUT:", OUT)
