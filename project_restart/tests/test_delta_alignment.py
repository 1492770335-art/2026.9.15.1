# -*- coding: utf-8 -*-
"""D-01 alignment unit tests for mlp_delta.make_delta / reconstruct_levels.
Invariant: reconstruct_levels(make_delta(y)) == y exactly (per day/code segment cumsum).
Also synthetic checks for day-start handling and cross-code/day isolation.
Run: python tests/test_delta_alignment.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "models"))

import polars as pl  # noqa: E402
from mlp_delta import make_delta, reconstruct_levels  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(("[PASS] " if cond else "[FAIL] ") + name + ("" if cond else f" {detail}"))
    if not cond:
        FAIL.append(name)


# --- synthetic: 2 dates x 3 ticks x 50 codes ---
rng = np.random.default_rng(0)
n_dates, n_ticks, n_codes = 2, 3, 50
N = n_dates * n_ticks * n_codes
y = rng.normal(size=(N, 1)).astype(np.float32)
dates = np.repeat([0, 1], n_ticks * n_codes)

dy = make_delta(y, dates)
lev = reconstruct_levels(dy[:, 0], dates)

# 1) identity invariant
check("reconstruct(make_delta(y)) == y", np.allclose(lev, y[:, 0], atol=1e-5), f"maxdiff={np.abs(lev-y[:,0]).max()}")

# 2) day-start rows: delta == y (y_{-1}=0 definition); first tick rows of each day = rows 0 and 150
day_start_rows = np.array([0, n_ticks * n_codes])
check("day-start delta == y", np.allclose(dy[day_start_rows, 0], y[day_start_rows, 0]),
      f"dy={dy[day_start_rows,0]} y={y[day_start_rows,0]}")

# 3) mid-day row: delta == y_t - y_{t-1} same code (row 50 -> code 0, tick 1 of day 0)
t = 50
check("mid-day delta == y_t - y_{t-1} (same code)",
      np.isclose(dy[t, 0], y[t, 0] - y[t - 50, 0]),
      f"dy={dy[t,0]} expect={y[t,0]-y[t-50,0]}")

# 4) cross-day isolation: last tick of day 0 vs first tick of day 1 (rows 100..149 and 150..199)
#    day1 first tick delta must NOT reference day0 last tick
t1 = n_ticks * n_codes  # 150, first row of day 1
check("no cross-day chaining", np.isclose(dy[t1, 0], y[t1, 0]) and not np.isclose(dy[t1, 0], y[t1, 0] - y[t1 - 50, 0], atol=1e-3),
      f"dy={dy[t1,0]} y={y[t1,0]}")

# --- real data full-scale identity check ---
DATA = os.path.join("project", "project", "data", "public.parquet")
frame = pl.scan_parquet(DATA).sort(["date", "time", "code"]).select(["date", "y0", "y1", "y2"]).collect()
y3 = frame.select(["y0", "y1", "y2"]).to_numpy().astype(np.float32)
dates_r = frame["date"].to_numpy()
ok_all = True
for j in range(3):
    dy = make_delta(y3[:, j:j + 1], dates_r)
    lev = reconstruct_levels(dy[:, 0], dates_r)
    md = np.abs(lev - y3[:, j]).max()
    ok_all &= md < 1e-4
    print(f"[INFO] real data target y{j}: max |reconstruct - y| = {md:.2e}")
check("real data identity holds for all 3 targets", ok_all)

print()
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL PASS")
