# -*- coding: utf-8 -*-
"""D-03 conditional incremental value (read-only, CPU, frozen EXP-0005 foldA weights):
1) strong-GRU level predictions on train(0-7)/val(8-11) via batched CPU inference
2) residual diagnostics: corr(e_t, f000_t) on train/val separately
3) train-only ridge of residuals on [f000] / [f000, delta_f003]; val corr with correction applied
4) ILLEGAL_FOR_SUBMISSION oracle: level reconstructed from TRUE deltas (upper bound, diagnostic only)
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
sys.path.insert(0, os.path.join("project_restart", "src", "models"))
from gru_strong import load_block, make_batch, CFG  # noqa: E402
from src.eval.metrics import pooled_pearson_raw, official_score  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
COMMIT = os.path.join("project_restart", "experiments", "EXP-0005", "commit_grustrong_foldA")
OUT = os.path.join("project_restart", "reports", "ph1_d03_residual_value.json")
t0 = time.time()
res = {}

# ---- load frozen models (read-only) ----
spec = importlib.util.spec_from_file_location("netw", os.path.join(COMMIT, "network.py"),
                                              submodule_search_locations=[os.path.abspath(COMMIT)])
netw = importlib.util.module_from_spec(spec)
sys.modules["netw"] = netw
spec.loader.exec_module(netw)
models = []
for label in range(3):
    cfg = json.load(open(os.path.join(COMMIT, "conf", f"Model_{label}.json"), encoding="utf-8"))
    m = getattr(netw, cfg["model"])(cfg).to("cpu")
    m.load_state_dict(torch.load(os.path.join(COMMIT, "model", f"Model_{label}.pth"), map_location="cpu", weights_only=True))
    m.eval()
    models.append(m)


def predict_levels(dates):
    """Batched predictions for the 3 frozen models over given dates -> (N,3)."""
    X, y, tid = load_block(dates)
    preds = []
    for m in models:
        o = []
        with torch.no_grad():
            for s in range(0, len(X), 4096):
                I = np.arange(s, min(s + 4096, len(X)))
                Xg = torch.from_numpy(make_batch(X, I, tid, CFG["window"])).to("cpu")
                o.append(m(Xg).numpy())
        preds.append(np.concatenate(o))
    return np.stack(preds, axis=1).reshape(-1, 3), y


def load_aux(dates):
    lf = pl.scan_parquet(DATA).filter(pl.col("date").is_in(sorted(dates))).sort(["date", "time", "code"])
    f = lf.select(["date", "f000", "f003", "y0", "y1", "y2"]).collect()
    d = f["date"].to_numpy()
    f000 = f["f000"].to_numpy().astype(np.float64)
    f003 = f["f003"].to_numpy().astype(np.float64)
    y = f.select(["y0", "y1", "y2"]).to_numpy().astype(np.float64)
    return d, f000, f003, y


def df003_of(f003, dates):
    n = len(f003)
    idx = np.arange(n)
    ok = (idx >= 50) & (dates == dates[np.clip(idx - 50, 0, n - 1)])
    out = np.zeros(n)
    out[ok] = f003[ok] - f003[np.clip(idx[ok] - 50, 0, n - 1)]
    return out


print("predicting train 0-7 ...", flush=True)
Ptr, Ytr = predict_levels(list(range(0, 8)))
print("predicting val 8-11 ...", flush=True)
Pva, Yva = predict_levels([8, 9, 10, 11])
dtr, f000_tr, f003_tr, _ = load_aux(list(range(0, 8)))
dva, f000_va, f003_va, _ = load_aux([8, 9, 10, 11])

base_val = {}
res["val_base_corr"] = {}
res["residual_corr_with_f000"] = {}
res["ridge_correction"] = {}
res["oracle_illegal_upper_bound"] = {}

for j, yn in enumerate(["y0", "y1", "y2"]):
    etr = Ytr[:, j] - Ptr[:, j]
    eva = Yva[:, j] - Pva[:, j]
    base = pooled_pearson_raw(Pva[:, j:j + 1], Yva[:, j:j + 1])[0]
    res["val_base_corr"][yn] = float(base)
    res["residual_corr_with_f000"][yn] = {
        "train": float(np.corrcoef(etr, f000_tr)[0, 1]),
        "val": float(np.corrcoef(eva, f000_va)[0, 1]),
    }
    # train-only ridge of residual on [f000] and [f000, df003]
    g_tr = df003_of(f003_tr, dtr)
    g_va = df003_of(f003_va, dva)
    for name, cols_tr, cols_va in (("f000", [f000_tr], [f000_va]),
                                   ("f000+df003", [f000_tr, g_tr], [f000_va, g_va])):
        A = np.stack([np.ones_like(cols_tr[0])] + cols_tr, 1)
        w, *_ = np.linalg.lstsq(A, etr, rcond=None)
        Av = np.stack([np.ones_like(cols_va[0])] + cols_va, 1)
        ehat = Av @ w
        c_corr = pooled_pearson_raw((Pva[:, j] + ehat)[:, None], Yva[:, j:j + 1])[0]
        res["ridge_correction"].setdefault(yn, {})[name] = {
            "val_corr_with_correction": float(c_corr),
            "delta_vs_base": float(c_corr - base),
        }
    # ILLEGAL oracle: reconstruct level from TRUE deltas (y_{-1}=0 day start)
    n = len(Yva)
    idx = np.arange(n)
    ok = (idx >= 50) & (dva == dva[np.clip(idx - 50, 0, n - 1)])
    dtrue = Yva[:, j].copy()
    dtrue[ok] = Yva[ok, j] - Yva[np.clip(idx[ok] - 50, 0, n - 1), j]
    # reuse matrix-view reconstruction
    L = dtrue.reshape(-1, 50).astype(np.float64)
    T = L.shape[0]
    tick_day = dva.reshape(-1, 50)[:, 0]
    changed = np.r_[True, tick_day[1:] != tick_day[:-1]]
    dst = np.maximum.accumulate(np.where(changed, np.arange(T), 0))
    cs = np.cumsum(L, axis=0)
    off = np.zeros_like(L)
    v = dst > 0
    off[v] = cs[dst[v] - 1]
    oracle = (cs - off).reshape(-1)
    o_corr = pooled_pearson_raw(oracle[:, None], Yva[:, j:j + 1])[0]
    res["oracle_illegal_upper_bound"][yn] = float(o_corr)

res["elapsed_sec"] = round(time.time() - t0, 1)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(res, f, indent=2, default=str)
print(json.dumps(res, indent=2, default=str))
