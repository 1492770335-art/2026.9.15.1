# -*- coding: utf-8 -*-
"""Delta-predictability for all three targets (chunked float32). Complements y2_ceiling.py."""
import json
import os
import time

import numpy as np
import polars as pl

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "reports", "ph1_delta_all_targets.json")
t0 = time.time()

lf = pl.scan_parquet(DATA)
feats = sorted([n for n in lf.collect_schema().names() if n.startswith("f")], key=lambda n: int(n[1:]))
frame = lf.sort(["date", "time", "code"]).select(["date", "y0", "y1", "y2"] + feats).collect()
X = frame.select(feats).to_numpy().astype(np.float32)
Y = frame.select(["y0", "y1", "y2"]).to_numpy().astype(np.float32)
dates = frame["date"].to_numpy()
n = len(X)
CH = 1_000_000

res = {}
for j, yn in enumerate(["y0", "y1", "y2"]):
    y = Y[:, j]
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
        dy = y[idx[ok]] - y[prev]
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
    res[yn] = {
        "top_f": {feats[i]: float(c1[i]) for i in np.argsort(-np.abs(c1))[:8]},
        "top_df": {feats[i]: float(c2[i]) for i in np.argsort(-np.abs(c2))[:8]},
        "max_abs_f": float(np.abs(c1).max()),
        "max_abs_df": float(np.abs(c2).max()),
    }
res["elapsed_sec"] = round(time.time() - t0, 1)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, default=str)
print(json.dumps(res, indent=2, default=str))
