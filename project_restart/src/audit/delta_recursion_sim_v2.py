# -*- coding: utf-8 -*-
"""D-03b 勘误版诊断 (v2)：修正真值锚定实现。
A  (legal):              yhat_0 = dhat_0;  yhat_t = yhat_{t-1} + dhat_t      (纯递归)
B2 (ILLEGAL, 修正版):     yhat_0 = y_0(true); yhat_t = y_0(true) + sum_{s=1..t} dhat_s
                         = levA + (y_0 - dhat_0) 在 (day,code) 内广播（锚点传播）
C  (ILLEGAL, 每tick真值):  yhat_t = y_{t-1}(true) + dhat_t (t>=1), yhat_0 = dhat_0
                         误差 = 单步 e_t，无积累 → 隔离"积累是否致崩坏"
v1 实现的 B（delta_recursion_sim.py L65-68）只在日初 tick 覆盖真实值、不传播锚点 → 已勘误。
"""
import json
import os
import time

import numpy as np
import polars as pl

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "reports", "ph1_d03b_v2_corrected_anchor.json")
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
first = np.r_[True, dva[1:] != dva[:-1]]
# day-start value lookup per row: propagate first-tick values via segment start mapping
changed_idx = np.where(first)[0]
seg_of = np.cumsum(first) - 1
start_idx = changed_idx[seg_of]

for j, yn in enumerate(["y0", "y1", "y2"]):
    dtrue_tr = delta_of(ytr[:, j], dtr)
    dtrue_va = delta_of(yva[:, j], dva)
    A = np.stack([np.ones_like(f000_tr), f000_tr], 1)
    w, *_ = np.linalg.lstsq(A, dtrue_tr, rcond=None)
    dhat_va = w[0] + w[1] * f000_va
    c_in = float(np.corrcoef(dhat_va, dtrue_va)[0, 1])
    levA = reconstruct(dhat_va, dva)
    # B2: corrected propagated day-start anchor (ILLEGAL)
    err0 = yva[start_idx, j] - dhat_va[start_idx]  # per-row day-start error (y_0 - dhat_0)
    levB2 = levA + err0
    # C: per-tick true previous label (ILLEGAL)
    levC = yva[np.clip(np.arange(len(dva)) - 50, 0, len(dva) - 1), j].copy()
    levC[first] = dhat_va[first]  # t=0: yhat_0 = dhat_0 (no true label at t-1)
    levC[~first] = levC[~first] + dhat_va[~first]
    res[yn] = {
        "val_corr_dhat_with_dtrue": c_in,
        "A_legal_pure_recursion": float(np.corrcoef(levA, yva[:, j])[0, 1]),
        "B2_illegal_daystart_anchor_propagated": float(np.corrcoef(levB2, yva[:, j])[0, 1]),
        "C_illegal_per_tick_true_prev_label": float(np.corrcoef(levC, yva[:, j])[0, 1]),
        "frozen_gru": [0.4518, 0.285, 0.0938][j],
    }
    print(json.dumps({yn: res[yn]}, indent=2), flush=True)

res["note"] = ("B2/C 为 ILLEGAL_FOR_SUBMISSION 离线诊断（B2 日初真实 y_0 且锚点传播；"
               "C 每 tick 使用真实 y_{t-1}）；A 为合法纯递归。均非训练、非提交方案。")
res["v1_erratum"] = ("delta_recursion_sim.py L65-68 的 B 变体仅在日初 tick 覆盖真实值且未传播锚点，"
                     "导致 B≈A 是代码伪象；本 v2 修正 B2 并新增 C。")
res["elapsed_sec"] = round(time.time() - t0, 1)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, default=str)
print("OUT:", OUT)
