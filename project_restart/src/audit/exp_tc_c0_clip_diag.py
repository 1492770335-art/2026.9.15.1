# -*- coding: utf-8 -*-
"""TC Gate C0 task 2: Clip hypothesis zero-training diagnosis.
Analyze f0-f82 continuous feature extremes/quantiles/tails/drift on inner_train vs outer_val
(per fold), using ONLY inner_train for any statistic that would drive a threshold. No training.
Distinguish input-feature clipping from existing gradient clipping=1.0.
"""
import json
import os
import sys

import numpy as np
import polars as pl

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0026")
os.makedirs(OUT, exist_ok=True)

CONT = [f"f{i:03d}" for i in range(83)]  # f000..f082 continuous


def stats(dates):
    df = pl.scan_parquet(DATA).filter(pl.col("date").is_in(sorted(dates))).select(CONT).collect().to_numpy().astype(np.float32)
    a = np.abs(df)
    q = np.percentile(a, [50, 90, 99, 99.9, 99.99, 100], axis=0)
    tail = {
        "gt3": float((a > 3).mean()), "gt5": float((a > 5).mean()),
        "gt10": float((a > 10).mean()), "gt20": float((a > 20).mean()),
    }
    return {
        "n_rows": int(len(df)),
        "mean_abs": float(a.mean()),
        "max_abs": float(a.max()),
        "per_feature_max_abs_p99": float(np.percentile(a.max(axis=0), 99)),
        "abs_quantiles_across_all": {"p50": float(np.percentile(a, 50)), "p99": float(np.percentile(a, 99)),
                                     "p999": float(np.percentile(a, 99.9)), "p9999": float(np.percentile(a, 99.99)),
                                     "pmax": float(a.max())},
        "tail_proportion": tail,
        "n_features_max_abs_gt10": int((a.max(axis=0) > 10).sum()),
        "n_features_max_abs_gt20": int((a.max(axis=0) > 20).sum()),
        "mean_mean": float(df.mean(axis=0).mean()), "std_mean": float(df.std(axis=0).mean()),
    }


if __name__ == "__main__":
    res = {
        "continuous_features": "f0..f82 (83 cols, float16 in parquet -> float32)",
        "folds": {},
        "existing_gradient_clip": 1.0,
        "note": "input feature clipping (hypothesis) is distinct from gradient clipping=1.0 (already in training).",
    }
    for fold, inner_train, inner_stop, outer_val in [
        ("A", list(range(0, 6)), [6, 7], [8, 9, 10, 11]),
        ("B", list(range(0, 10)), [10, 11], [12, 13, 14, 15]),
    ]:
        res["folds"][fold] = {
            "inner_train": stats(inner_train),
            "inner_stop": stats(inner_stop),
            "outer_val": stats(outer_val),
        }
    with open(os.path.join(OUT, "clip_diagnosis.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    # concise summary
    for fold in ("A", "B"):
        it = res["folds"][fold]["inner_train"]
        ov = res["folds"][fold]["outer_val"]
        print(f"[{fold}] inner_train max_abs={it['max_abs']:.3f} p999={it['abs_quantiles_across_all']['p999']:.3f} "
              f"tail>5={it['tail_proportion']['gt5']:.6f} tail>10={it['tail_proportion']['gt10']:.6f} "
              f"feat_max>10={it['n_features_max_abs_gt10']}/83; outer max_abs={ov['max_abs']:.3f} p999={ov['abs_quantiles_across_all']['p999']:.3f}", flush=True)
    print("done")
