# -*- coding: utf-8 -*-
"""Offline ceiling analysis for y2 (EXP-0013 premise gate):
1) persistence ceiling: corr(y2_t, y2_{t-k}) for k=1..10 (per stock, pooled across days 0-19)
2) delta predictability: per-feature corr(Delta y2_t, f_t) and corr(Delta y2_t, Delta f_t)
Run: python src/audit/y2_ceiling.py
"""
import json
import os
import time

import numpy as np
import polars as pl

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "reports", "ph1_y2_ceiling.json")
t0 = time.time()
res = {}

lf = pl.scan_parquet(DATA)
feats = sorted([n for n in lf.collect_schema().names() if n.startswith("f")], key=lambda n: int(n[1:]))

# --- persistence ceiling (per stock, pooled over days) ---
frame = lf.sort(["date", "time", "code"]).select(["date", "code", "y0", "y1", "y2"] + feats).collect()
y2 = frame["y2"].to_numpy().astype(np.float64)   # (N,) ordered (date,time,code)
y0 = frame["y0"].to_numpy().astype(np.float64)
y1 = frame["y1"].to_numpy().astype(np.float64)
X = frame.select(feats).to_numpy().astype(np.float64)
dates = frame["date"].to_numpy()

pers = {}
for k in [1, 2, 3, 5, 10, 25, 50]:
    cc = []
    for yn, y in [("y0", y0), ("y1", y1), ("y2", y2)]:
        mask = np.ones(len(y), dtype=bool)
        # within-day only: lag k valid when same date and same code
        same = np.zeros(len(y), dtype=bool)
        same[k:] = (dates[k:] == dates[:-k]) & (np.arange(len(y))[k:] % 50 == np.arange(len(y))[:-k] % 50)
        cur = y[same]
        prev = y[np.roll(same, k) & same] if False else None
        # simpler: shift by k*50 rows (same code k ticks earlier) and require same date
        idx = np.arange(len(y))
        lag_idx = idx - k * 50
        ok = lag_idx >= 0
        ok &= dates == dates[np.clip(lag_idx, 0, len(y) - 1)]
        cur = y[ok]
        prev = y[np.clip(lag_idx[ok], 0, len(y) - 1)]
        c = np.corrcoef(cur, prev)[0, 1]
        cc.append(c)
    pers[k] = {"y0": cc[0], "y1": cc[1], "y2": cc[2]}
res["persistence_lag_corr"] = pers

# --- delta predictability (chunked float32 accumulation) ---
X = frame.select(feats).to_numpy().astype(np.float32)
dates = frame["date"].to_numpy()
y2f = y2.astype(np.float32)
n = len(X)
CH = 1_000_000
s_x = np.zeros(100, dtype=np.float64)
s_dx = np.zeros(100, dtype=np.float64)
s_dy = 0.0
s_x2 = np.zeros(100, dtype=np.float64)
s_dx2 = np.zeros(100, dtype=np.float64)
s_dy2 = 0.0
s_xdy = np.zeros(100, dtype=np.float64)
s_dxdy = np.zeros(100, dtype=np.float64)
cnt = 0
for s0 in range(0, n, CH):
    e = min(n, s0 + CH)
    idx = np.arange(s0, e)
    ok = idx >= 50
    ok &= dates[idx] == dates[np.clip(idx - 50, 0, n - 1)]
    if not ok.any():
        continue
    prev = np.clip(idx[ok] - 50, 0, n - 1)
    dy = y2f[idx[ok]] - y2f[prev]
    Xl = X[idx[ok]].astype(np.float64)
    dXl = (X[idx[ok]] - X[prev]).astype(np.float64)
    cnt += len(dy)
    s_x += Xl.sum(0)
    s_dx += dXl.sum(0)
    s_dy += dy.sum()
    s_x2 += (Xl * Xl).sum(0)
    s_dx2 += (dXl * dXl).sum(0)
    s_dy2 += float((dy * dy).sum())
    s_xdy += (Xl * dy[:, None]).sum(0)
    s_dxdy += (dXl * dy[:, None]).sum(0)
mx = s_x / cnt
mdx = s_dx / cnt
mdy = s_dy / cnt
cov_xdy = s_xdy / cnt - mx * mdy
cov_dxdy = s_dxdy / cnt - mdx * mdy
var_x = s_x2 / cnt - mx * mx
var_dx = s_dx2 / cnt - mdx * mdx
var_dy = s_dy2 / cnt - mdy * mdy
c1 = cov_xdy / np.sqrt(var_x * var_dy)
c2 = cov_dxdy / np.sqrt(var_dx * var_dy)
res["corr_dy2_with_f_top10"] = {feats[i]: float(c1[i]) for i in np.argsort(-np.abs(c1))[:10]}
res["corr_dy2_with_df_top10"] = {feats[i]: float(c2[i]) for i in np.argsort(-np.abs(c2))[:10]}
res["max_abs"] = {"dy2_f": float(np.abs(c1).max()), "dy2_df": float(np.abs(c2).max())}
res["elapsed_sec"] = round(time.time() - t0, 1)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, default=str)
print(json.dumps({k: res[k] for k in ("persistence_lag_corr", "corr_dy2_with_f_top10", "corr_dy2_with_df_top10", "max_abs", "elapsed_sec")}, indent=2, default=str))
print("OUT:", OUT)
