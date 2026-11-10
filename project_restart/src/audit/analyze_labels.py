# -*- coding: utf-8 -*-
"""Light CPU analysis: label autocorrelation (cross-day same tick & within-day lag-1) and
feature-target linear correlations. Informs horizon/leakage assessment and P1 feature choices.
"""
import json
import os
import time

import numpy as np
import polars as pl

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "reports", "ph1_label_feature_analysis.json")

t0 = time.time()
res = {}

lf = pl.scan_parquet(DATA)
feats = sorted([n for n in lf.collect_schema().names() if n.startswith("f")], key=lambda n: int(n[1:]))

# 1) feature-target pooled linear correlations (float32)
frame = lf.sort(["date", "time", "code"]).select(feats + ["y0", "y1", "y2"]).collect()
X = frame.select(feats).to_numpy().astype(np.float64)
Y = frame.select(["y0", "y1", "y2"]).to_numpy().astype(np.float64)
Xc = X - X.mean(0)
Yc = Y - Y.mean(0)
denom = np.sqrt((Xc ** 2).sum(0)[:, None] * (Yc ** 2).sum(0)[None, :])
corr = (Xc.T @ Yc) / denom
top = {}
for j, yn in enumerate(["y0", "y1", "y2"]):
    idx = np.argsort(-np.abs(corr[:, j]))[:10]
    top[yn] = {feats[i]: float(corr[i, j]) for i in idx}
res["feature_target_corr_top10"] = top
res["feature_target_corr_extreme"] = {
    "max_abs": float(np.abs(corr).max()),
    "where": [feats[int(i)] for i in np.where(np.abs(corr) > 0.1)[0]][:40],
}
del frame, X, Y, Xc, Yc

# 2) cross-day same-(time,code) label correlation: join day d with day d+1 on (time, code)
df = lf.select(["date", "time", "code", "y0", "y1", "y2"]).collect()
cross_day = []
for d in range(19):
    a = df.filter(pl.col("date") == d)
    b = df.filter(pl.col("date") == d + 1).select(["time", "code", "y0", "y1", "y2"]).rename(
        {"y0": "y0n", "y1": "y1n", "y2": "y2n"})
    j = a.join(b, on=["time", "code"], how="inner")
    row = {"pair": f"{d}-{d+1}", "n": j.height}
    for yn in ["y0", "y1", "y2"]:
        x = j[yn].to_numpy().astype(np.float64)
        y = j[yn + "n"].to_numpy().astype(np.float64)
        row[f"corr_{yn}"] = float(np.corrcoef(x, y)[0, 1])
    cross_day.append(row)
res["cross_day_same_tick_label_corr"] = cross_day

# 3) within-day lag-1 autocorrelation of labels (per stock, average over stocks)
within_day = {}
for d in [0, 8, 16]:
    a = df.filter(pl.col("date") == d).sort(["time", "code"])
    y = a.select(["y0", "y1", "y2"]).to_numpy().astype(np.float64)  # (ticks*50, 3) ordered time,code
    y = y.reshape(-1, 50, 3)
    lag = {}
    for j, yn in enumerate(["y0", "y1", "y2"]):
        cur = y[1:, :, j]
        prev = y[:-1, :, j]
        c = np.corrcoef(cur.ravel(), prev.ravel())[0, 1]
        lag[yn] = float(c)
    within_day[int(d)] = lag
res["within_day_lag1_label_corr"] = within_day

# 4) per-date label mean drift (sanity: any systematic cross-day trend)
res["per_date_label_mean"] = {
    int(d): {yn: float(df.filter(pl.col("date") == d)[yn].mean()) for yn in ["y0", "y1", "y2"]}
    for d in range(20)
}

res["elapsed_sec"] = round(time.time() - t0, 1)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, default=str)

print("elapsed", res["elapsed_sec"])
print("cross_day_same_tick_label_corr:")
for r in res["cross_day_same_tick_label_corr"]:
    print(" ", r["pair"], "n=", r["n"], "y0=%.3f y1=%.3f y2=%.3f" % (r["corr_y0"], r["corr_y1"], r["corr_y2"]))
print("within_day_lag1:", json.dumps(res["within_day_lag1_label_corr"]))
print("top features:")
for yn, t in res["feature_target_corr_top10"].items():
    print(" ", yn, json.dumps(t))
print("OUT:", OUT)
