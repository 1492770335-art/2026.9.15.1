# -*- coding: utf-8 -*-
"""D-02 delta signal audit (read-only, CPU):
- per-date corr(delta_y, f000) for dates 0-15 and grouped by A/B train/val folds
- cross-lag corr(delta_y_t, f000_{t+k}) k=-2..+2 (k>0 = FUTURE features: ILLEGAL_FOR_SUBMISSION,
  offline diagnostic only)
- per-code sign consistency
- redundancy: corr(f000, delta_f003); partial correlations
- robustness: extremes removed (diagnostic only, no data modification)
"""
import json
import os
import time

import numpy as np
import polars as pl

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "reports", "ph1_d02_delta_audit.json")
t0 = time.time()
res = {}

lf = pl.scan_parquet(DATA).sort(["date", "time", "code"])
frame = lf.select(["date", "f000", "f003", "y0", "y1", "y2"]).collect()
dates = frame["date"].to_numpy()
f000 = frame["f000"].to_numpy().astype(np.float64)
f003 = frame["f003"].to_numpy().astype(np.float64)
Y = frame.select(["y0", "y1", "y2"]).to_numpy().astype(np.float64)
n = len(Y)
res["n_rows"] = n


def lag_series(a, k, dates):
    """a shifted by k ticks (k*50 rows); invalid where out-of-date."""
    idx = np.arange(len(a))
    src = idx - k * 50
    ok = (src >= 0) & (src < len(a)) & (dates == dates[np.clip(src, 0, len(a) - 1)])
    out = np.full(len(a), np.nan)
    out[ok] = a[np.clip(src[ok], 0, len(a) - 1)]
    return out, ok


def corr_masked(x, y, ok):
    x = x[ok]
    y = y[ok]
    return float(np.corrcoef(x, y)[0, 1])


# --- per-date corr(delta y_j, f000) ---
per_date = {}
for d in range(16):  # restrict to 0-15 (A/B regions; 16-19 already historically exposed)
    m = dates == d
    # delta y within day (tick 0 excluded, y_{-1}=0 not used for corr on tick>=1)
    idx = np.arange(n)[m]
    prev_ok = (idx >= 50) & (dates[np.clip(idx - 50, 0, n - 1)] == d)
    dy = {j: (Y[idx[prev_ok], j] - Y[np.clip(idx[prev_ok] - 50, 0, n - 1), j]) for j in range(3)}
    f = f000[idx[prev_ok]]
    per_date[int(d)] = {f"y{j}": corr_masked(dy[j], f, np.ones(len(dy[j]), bool)) for j in range(3)}
res["per_date_corr_dy_f000"] = per_date

# --- grouped by fold regions ---
regions = {
    "A_train_0_7": list(range(0, 8)),
    "A_val_8_11": list(range(8, 12)),
    "B_train_0_11": list(range(0, 12)),
    "B_val_12_15": list(range(12, 16)),
}
res["region_corr_dy_f000"] = {}
for rname, ds in regions.items():
    m = np.isin(dates, ds)
    idx = np.arange(n)[m]
    prev_ok = (idx >= 50) & np.isin(dates[np.clip(idx - 50, 0, n - 1)], ds) & (dates[idx] == dates[np.clip(idx - 50, 0, n - 1)])
    row = {}
    for j in range(3):
        dy = Y[idx[prev_ok], j] - Y[np.clip(idx[prev_ok] - 50, 0, n - 1), j]
        row[f"y{j}"] = corr_masked(dy, f000[idx[prev_ok]], np.ones(len(dy), bool))
    res["region_corr_dy_f000"][rname] = row

# --- cross-lag: corr(delta_y_t, f000_{t+k}) within day, k=-2..2 ---
cross = {}
for j in range(3):
    # delta y at tick t (rows idx), valid when idx-50 same date
    ok_dy = (np.arange(n) >= 50) & (dates == dates[np.clip(np.arange(n) - 50, 0, n - 1)])
    dy = np.full(n, np.nan)
    dy[ok_dy] = Y[ok_dy, j] - Y[np.clip(np.arange(n)[ok_dy] - 50, 0, n - 1), j]
    m15 = np.isin(dates, list(range(0, 16)))
    cross[f"y{j}"] = {}
    for k in range(-2, 3):
        fk, okf = lag_series(f000, k, dates)
        ok = ok_dy & okf & m15
        cross[f"y{j}"][f"k={k}"] = corr_masked(dy, fk, ok)
res["cross_lag_corr_dy_f000"] = cross
res["cross_lag_note"] = "k>0 为未来特征，仅离线诊断，ILLEGAL_FOR_SUBMISSION"

# --- per-code sign consistency (dates 0-15) ---
sign = {}
for j in range(3):
    ok_dy = (np.arange(n) >= 50) & (dates == dates[np.clip(np.arange(n) - 50, 0, n - 1)]) & np.isin(dates, list(range(0, 16)))
    dy = Y[ok_dy, j] - Y[np.clip(np.arange(n)[ok_dy] - 50, 0, n - 1), j]
    f = f000[ok_dy]
    codes = (np.arange(n) % 50)[ok_dy]
    cc = []
    for c in range(50):
        m = codes == c
        if m.sum() > 100:
            cc.append(float(np.corrcoef(f[m], dy[m])[0, 1]))
    cc = np.array(cc)
    sign[f"y{j}"] = {"n_codes": len(cc), "min": float(cc.min()), "median": float(np.median(cc)), "max": float(cc.max()),
                     "negative_fraction": float((cc < 0).mean())}
res["per_code_corr_dy_f000"] = sign

# --- redundancy: corr(f000, delta_f003), partial corr ---
ok_d = (np.arange(n) >= 50) & (dates == dates[np.clip(np.arange(n) - 50, 0, n - 1)]) & np.isin(dates, list(range(0, 16)))
df003 = f003 - f003[np.clip(np.arange(n) - 50, 0, n - 1)]
res["corr_f000_df003"] = float(np.corrcoef(f000[ok_d], df003[ok_d])[0, 1])
res["partial_corr"] = {}
for j in range(3):
    dy = Y[ok_d, j] - Y[np.clip(np.arange(n)[ok_d] - 50, 0, n - 1), j]
    f = f000[ok_d]
    g = df003[ok_d]
    # partial corr(dy, f | g) and (dy, g | f) via linear regression residuals
    from numpy.linalg import lstsq
    A = np.stack([np.ones_like(g), g], 1)
    rf = f - A @ lstsq(A, f, rcond=None)[0]
    rg = g - A[:, 0] - 0 * g  # placeholder
    A2 = np.stack([np.ones_like(f), f], 1)
    rg = g - A2 @ lstsq(A2, g, rcond=None)[0]
    rdy_f = dy - A @ lstsq(A, dy, rcond=None)[0]
    rdy_g = dy - A2 @ lstsq(A2, dy, rcond=None)[0]
    res["partial_corr"][f"y{j}"] = {
        "dy_f_given_df003": float(np.corrcoef(rdy_f, rf)[0, 1]),
        "dy_df003_given_f000": float(np.corrcoef(rdy_g, rg)[0, 1]),
    }

# --- robustness: extremes removed (diagnostic only) ---
rob = {}
for j in range(3):
    ok_dy = (np.arange(n) >= 50) & (dates == dates[np.clip(np.arange(n) - 50, 0, n - 1)]) & np.isin(dates, list(range(0, 16)))
    dy = Y[ok_dy, j] - Y[np.clip(np.arange(n)[ok_dy] - 50, 0, n - 1), j]
    f = f000[ok_dy]
    base = float(np.corrcoef(f, dy)[0, 1])
    m = (np.abs(dy - dy.mean()) < 3 * dy.std()) & (np.abs(f - f.mean()) < 3 * f.std())
    trimmed = float(np.corrcoef(f[m], dy[m])[0, 1])
    rob[f"y{j}"] = {"full": base, "3sigma_trimmed": trimmed}
res["robustness"] = rob

res["elapsed_sec"] = round(time.time() - t0, 1)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, default=str)
print(json.dumps({k: res[k] for k in ("per_date_corr_dy_f000", "region_corr_dy_f000", "cross_lag_corr_dy_f000", "per_code_corr_dy_f000", "corr_f000_df003", "partial_corr", "robustness", "elapsed_sec")}, indent=2, default=str))
