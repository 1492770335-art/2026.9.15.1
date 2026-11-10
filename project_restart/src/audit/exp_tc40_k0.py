# -*- coding: utf-8 -*-
"""TC-40 K0 category audit: verify f83-f99 values in {0,1,2}, frequency/missing/rare per fold region,
current encoding, plus resource estimate (GRU128 100->134) and baseline hashes. Zero-training.
"""
import hashlib
import json
import os
import sys

import numpy as np
import polars as pl

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0028")
os.makedirs(OUT, exist_ok=True)
CAT = [f"f{i:03d}" for i in range(83, 100)]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def audit(dates):
    df = pl.scan_parquet(DATA).filter(pl.col("date").is_in(sorted(dates))).select(CAT).collect().to_numpy().astype(np.int64)
    out = {}
    for j, col in enumerate(CAT):
        vals = df[:, j]
        uniq = sorted(set(vals.tolist()))
        out[col] = {"unique": uniq,
                    "count0": int((vals == 0).sum()), "count1": int((vals == 1).sum()), "count2": int((vals == 2).sum()),
                    "frac0": float((vals == 0).mean()), "frac1": float((vals == 1).mean()), "frac2": float((vals == 2).mean()),
                    "outside_012": int((~np.isin(vals, [0, 1, 2])).sum())}
    return out, df.shape[0]


if __name__ == "__main__":
    res = {"data_sha256": sha256(DATA), "evaluate_py_sha256": sha256(os.path.join("project", "project", "script", "evaluate.py")),
           "exp0016_y2_baseline_sha": {
               "A": sha256(os.path.join("project_restart", "experiments", "EXP-0016", "commit_K16_A", "model", "Model_2.pth")),
               "B": sha256(os.path.join("project_restart", "experiments", "EXP-0016", "commit_K16_B", "model", "Model_2.pth")),
           },
           "exp0016_raw_c2": {"A": 0.125503, "B": 0.170568},
           "current_encoding": "K16-GRU 网络 nn.GRU(100,128)，f083-f099 作为数值 0/1/2 直接输入（序数/无 one-hot/无 embedding）",
           "folds": {}}
    for fold, inner_fit, inner_stop in [("A", list(range(0, 6)), [6, 7]), ("B", list(range(0, 10)), [10, 11])]:
        fi, n_fi = audit(inner_fit)
        fs, n_fs = audit(inner_stop)
        res["folds"][fold] = {"inner_fit": {"n_rows": n_fi, "per_col": fi},
                              "inner_stop": {"n_rows": n_fs, "per_col": fs}}
    # global category value check across all 20 dates
    allv = pl.scan_parquet(DATA).select(CAT).collect().to_numpy().astype(np.int64)
    outside = int((~np.isin(allv, [0, 1, 2])).sum())
    res["global_outside_012_count"] = outside
    res["global_total_cells"] = int(allv.size)
    res["n_unique_per_col_global"] = {CAT[j]: sorted(set(allv[:, j].tolist())) for j in range(17)}
    # rare categories: min frac per state across all data
    rare = {}
    for j, col in enumerate(CAT):
        v = allv[:, j]
        rare[col] = {int(s): float((v == s).mean()) for s in (0, 1, 2)}
    res["global_state_fraction_per_col"] = rare
    # resource estimate
    res["resource_estimate"] = {
        "input_dim": {"baseline": 100, "onehot": 134},
        "gru128_wih_params": {"baseline": 3 * 128 * 100, "onehot": 3 * 128 * 134, "delta": 3 * 128 * 34},
        "data_memory_ratio": 134 / 100,
        "note": "GRU 仅输入->隐层权重增长 3*128*34=13056 参数；存在输入维数/容量混淆，若 K1 有增益需等容量 placebo 控制",
    }
    with open(os.path.join(OUT, "category_audit.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    print("global_outside_012:", outside, "/", res["global_total_cells"])
    print("n_unique_per_col_global:", res["n_unique_per_col_global"])
    print("resource delta params:", res["resource_estimate"]["gru128_wih_params"]["delta"])
    print("done")
