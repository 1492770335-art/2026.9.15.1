# -*- coding: utf-8 -*-
"""Ridge baseline: closed-form per-target ridge with intercept, chunked float64 accumulation.
Features: f000-f099 in float32 (same as official evaluator input). No re-standardization (features are
already standardized; pre-registered default lam=100). Deterministic, no seed.
"""
import os
import time

import numpy as np
import polars as pl


def load_block(data_path: str, dates, device_features=True):
    """Return X (N,100) float32, y (N,3) float64, keys (N,4) int64 in (date,time,code) order."""
    lf = pl.scan_parquet(data_path).filter(pl.col("date").is_in(sorted(dates)))
    frame = lf.sort(["date", "time", "code"]).collect()
    feats = sorted([n for n in frame.columns if n.startswith("f")], key=lambda n: int(n[1:]))
    X = frame.select(feats).to_numpy().astype(np.float32)
    y = frame.select(["y0", "y1", "y2"]).to_numpy().astype(np.float64)
    keys = frame.select(["date", "time", "code"]).to_numpy().astype(np.int64)
    return X, y, keys


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge with intercept. X: (N,F) float32, y: (N,T) float64 -> W: (F+1, T)."""
    X = X.astype(np.float64)
    N, F = X.shape
    XtX = np.zeros((F + 1, F + 1), dtype=np.float64)
    Xty = np.zeros((F + 1, y.shape[1]), dtype=np.float64)
    CH = 1 << 20
    for s in range(0, N, CH):
        Xc = np.concatenate([X[s:s + CH], np.ones((X[s:s + CH].shape[0], 1))], axis=1)
        XtX += Xc.T @ Xc
        Xty += Xc.T @ y[s:s + CH]
    XtX[0:F, 0:F] += lam * np.eye(F)
    W = np.linalg.solve(XtX, Xty)
    return W


def predict_ridge(X: np.ndarray, W: np.ndarray) -> np.ndarray:
    X = X.astype(np.float64)
    return np.concatenate([X, np.ones((X.shape[0], 1))], axis=1) @ W


if __name__ == "__main__":
    t0 = time.time()
    DATA = os.path.join("project", "project", "data", "public.parquet")
    OUT = os.path.join("project_restart", "experiments", "EXP-0001")
    os.makedirs(OUT, exist_ok=True)

    lams = [1.0, 10.0, 100.0, 1000.0]
    tasks = {
        "foldA": {"train": list(range(0, 8)), "val": list(range(8, 12))},
        "foldB": {"train": list(range(0, 12)), "val": list(range(12, 16))},
        "gap0": {"train": list(range(0, 8)), "val": list(range(8, 12))},
        "gap1": {"train": list(range(0, 7)), "val": list(range(8, 12))},
    }
    results = {}
    import json
    import sys
    sys.path.insert(0, os.path.join("project_restart"))
    from src.eval.metrics import official_score, pooled_pearson_raw

    for task, spec in tasks.items():
        Xtr, ytr, _ = load_block(DATA, spec["train"])
        Xva, yva, kva = load_block(DATA, spec["val"])
        for lam in lams:
            W = fit_ridge(Xtr, ytr, lam)
            p = predict_ridge(Xva, W)
            o = official_score(p, yva)
            c = pooled_pearson_raw(p, yva)
            # per-day diagnostics
            daily = {}
            for d in spec["val"]:
                m = kva[:, 0] == d
                dd = official_score(p[m], yva[m])
                daily[int(d)] = dd
            key = f"{task}_lam{int(lam)}"
            results[key] = {"lam": lam, "train": spec["train"], "val": spec["val"],
                            "corr0": o["corr0"], "corr1": o["corr1"], "corr2": o["corr2"],
                            "raw_score": o["raw_score"], "official_score": o["score"],
                            "raw_corr": c.tolist(), "daily": daily}
            if task == "foldA" and lam == 100.0:
                # save OOF-style predictions for baseline lam=100, fold A
                np.save(os.path.join(OUT, "foldA_lam100_preds.npy"), p.astype(np.float32))
                np.save(os.path.join(OUT, "foldA_lam100_keys.npy"), kva)
                np.save(os.path.join(OUT, "foldA_lam100_y.npy"), yva)
                print(f"[saved] foldA lam=100 preds -> {OUT}")

    results["elapsed_sec"] = round(time.time() - t0, 1)
    with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)

    print(json.dumps({k: {kk: v[kk] for kk in ("corr0", "corr1", "corr2", "raw_score", "official_score")}
                      for k, v in results.items() if k != "elapsed_sec"}, indent=2))
    print("elapsed_sec:", results["elapsed_sec"])
