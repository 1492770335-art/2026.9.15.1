# -*- coding: utf-8 -*-
"""Phase 0 full-scan data audit for public.parquet (Turing Cup high-freq prediction).
Run:  python src/audit/audit_data.py   (cwd = 比赛)
Read-only. Outputs JSON + prints summary.
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import polars as pl

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "reports", "ph0_audit_results.json")

t0 = time.time()
results = {"data": DATA, "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z")}

with open(DATA, "rb") as f:
    h = hashlib.sha256()
    for chunk in iter(lambda: f.read(1 << 24), b""):
        h.update(chunk)
results["data_sha256"] = h.hexdigest()
results["data_size_bytes"] = os.path.getsize(DATA)

lf = pl.scan_parquet(DATA)
schema = lf.collect_schema()
names = schema.names()
results["n_columns"] = len(names)
results["columns"] = names
results["dtypes"] = {n: str(schema.dtypes()[i]) for i, n in enumerate(names)}

n_rows = lf.select(pl.len()).collect().item()
results["n_rows"] = n_rows
results["n_unique_keys"] = lf.select(pl.struct(["date", "time", "code"]).n_unique()).collect().item()

# --- per-date aggregates ---
per_date = (
    lf.group_by("date")
    .agg(
        pl.len().alias("rows"),
        pl.col("time").n_unique().alias("ticks"),
        pl.col("time").min().alias("t_min"),
        pl.col("time").max().alias("t_max"),
        pl.col("code").n_unique().alias("codes"),
        pl.col("code").min().alias("code_min"),
        pl.col("code").max().alias("code_max"),
    )
    .sort("date")
    .collect()
)
results["per_date"] = per_date.to_dicts()

# --- every tick must be exactly 50 rows with 50 distinct codes ---
bad_ticks = (
    lf.group_by(["date", "time"])
    .agg(pl.len().alias("n"), pl.col("code").n_unique().alias("nc"))
    .filter((pl.col("n") != 50) | (pl.col("nc") != 50))
    .collect()
)
results["bad_tick_rows"] = bad_ticks.height
results["bad_ticks_sample"] = bad_ticks.head(10).to_dicts()

# --- duplicate keys ---
dups = (
    lf.group_by(["date", "time", "code"])
    .agg(pl.len().alias("n"))
    .filter(pl.col("n") > 1)
    .collect()
)
results["duplicate_keys"] = dups.height
results["duplicate_sample"] = dups.head(10).to_dicts()

# --- null counts (all columns) ---
nulls = lf.select(pl.all().null_count()).collect()
nc = {n: nulls[0, i] for i, n in enumerate(names)}
results["null_counts"] = nc

# --- NaN / Inf per float column ---
float_cols = [n for n in names if schema.dtypes()[names.index(n)] in (pl.Float16, pl.Float32, pl.Float64)]
exprs = []
for c in float_cols:
    exprs.append(pl.col(c).is_nan().sum().alias(c + "__nan"))
    exprs.append(pl.col(c).is_infinite().sum().alias(c + "__inf"))
fin = lf.select(exprs).collect().to_dicts()[0]
results["nan_counts"] = {k[:-6]: int(v) for k, v in fin.items() if k.endswith("__nan")}
results["inf_counts"] = {k[:-6]: int(v) for k, v in fin.items() if k.endswith("__inf")}

# --- category domain f083..f099 ---
cat_cols = [n for n in names if n.startswith("f") and 83 <= int(n[1:]) <= 99]
cat_exprs = []
for c in cat_cols:
    cat_exprs.append(pl.col(c).min().alias(c + "__min"))
    cat_exprs.append(pl.col(c).max().alias(c + "__max"))
    cat_exprs.append(pl.col(c).n_unique().alias(c + "__nuniq"))
cat_stats = lf.select(cat_exprs).collect().to_dicts()[0]
results["category_domain"] = {
    c: {"min": int(cat_stats[c + "__min"]), "max": int(cat_stats[c + "__max"]),
        "n_unique": int(cat_stats[c + "__nuniq"])}
    for c in cat_cols
}

# --- global extrema of continuous features & labels (min/max/quantiles) ---
for c in ["f000", "f001", "f082", "y0", "y1", "y2"]:
    row = (
        lf.select(
            pl.col(c).min().alias("min"),
            pl.col(c).max().alias("max"),
            pl.col(c).quantile(0.01).alias("q01"),
            pl.col(c).quantile(0.5).alias("q50"),
            pl.col(c).quantile(0.99).alias("q99"),
            pl.col(c).mean().alias("mean"),
            pl.col(c).std().alias("std"),
        ).collect()
    )
    results["global_" + c] = {k: float(row[k][0]) for k in ["min", "max", "q01", "q50", "q99", "mean", "std"]}

# --- per-date label stats ---
lab_exprs = []
for y in ["y0", "y1", "y2"]:
    for op in ["mean", "std", "min", "max"]:
        lab_exprs.append(getattr(pl.col(y), op)().alias(f"{y}__{op}"))
per_date_lab = lf.group_by("date").agg(lab_exprs).sort("date").collect()
results["per_date_labels"] = per_date_lab.to_dicts()

# --- time-order check inside each date (raw file order) ---
ord_frame = lf.select(["date", "time", "code"]).collect().to_numpy()
per_date_sorted_ok = {}
for d in range(20):
    m = ord_frame[:, 0] == d
    t = ord_frame[m, 1]
    per_date_sorted_ok[int(d)] = bool(np.all(np.diff(t.astype(np.int64)) >= 0))
results["raw_time_monotonic_per_date"] = per_date_sorted_ok

# --- time consecutive check: ticks == t_max - t_min + 1 ---
for row in per_date.to_dicts():
    row["time_consecutive"] = bool(row["ticks"] == row["t_max"] - row["t_min"] + 1)
results["per_date"] = per_date.to_dicts()

# --- code order within tick after official sort (time, code) ---
srt = lf.sort(["time", "code"]).select(["date", "code"]).collect().to_numpy()
codes_ok = True
offs = 0
for d in range(20):
    codes = srt[srt[:, 0] == d, 1]
    if not np.array_equal(codes, np.tile(np.arange(50, dtype=codes.dtype), len(codes) // 50)):
        codes_ok = False
        break
results["official_sort_code_sequence_ok"] = codes_ok

results["elapsed_sec"] = round(time.time() - t0, 1)
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2, default=str)

print(json.dumps({k: results[k] for k in [
    "data_sha256", "data_size_bytes", "n_columns", "n_rows", "n_unique_keys",
    "bad_tick_rows", "duplicate_keys", "official_sort_code_sequence_ok",
    "raw_time_monotonic_per_date", "elapsed_sec"]}, indent=2, default=str))
print("per_date:")
print(json.dumps(results["per_date"], indent=2))
print("category_domain:")
print(json.dumps(results["category_domain"], indent=2))
print("global f000/y0/y1/y2:")
for k in ["global_f000", "global_f082", "global_y0", "global_y1", "global_y2"]:
    print(k, json.dumps(results[k]))
print("null_counts total:", sum(results["null_counts"].values()))
print("nan_counts total:", sum(results["nan_counts"].values()))
print("inf_counts total:", sum(results["inf_counts"].values()))
print("per_date_labels:")
print(json.dumps(results["per_date_labels"], indent=2))
print("OUT:", OUT)
