# -*- coding: utf-8 -*-
"""Window alignment unit tests for K in {16,64,100} (make_batch from gru_strong):
1) left-padding zeros when history shorter than K; 2) causal content = current + previous ticks of SAME code;
3) no cross-day chaining (day-start row uses only zeros before its own first tick);
4) no cross-code contamination.
Run: python tests/test_window_alignment.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "models"))
from gru_strong import make_batch, CFG  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(("[PASS] " if cond else "[FAIL] ") + name + ("" if cond else f" {detail}"))
    if not cond:
        FAIL.append(name)


# synthetic: 2 dates x 3 ticks x 50 codes, feature = row index as float (unique, traceable)
n_dates, n_ticks, n_codes = 2, 3, 50
N = n_dates * n_ticks * n_codes
X = np.arange(N, dtype=np.float32).reshape(-1, 1).repeat(100, axis=1)  # each row's feature = its index
# tick_in_day: for a row i (ordered date,time,code), ticks since day start = (i - day_start_row)//50
dates = np.repeat([0, 1], n_ticks * n_codes)
changed = np.r_[True, dates[1:] != dates[:-1]]
day_start_row = np.maximum.accumulate(np.where(changed, np.arange(N), 0))
tick_in_day = (np.arange(N) - day_start_row).astype(np.int64) // 50

for K in (16, 64, 100):
    CFG["window"] = K
    # row of tick 2, code 7, day 0: index = 2*50 + 7 = 107
    i = 107
    Xg = make_batch(X, np.array([i]), tick_in_day, K)  # (1, K, 100)
    row = Xg[0, :, 0]
    # expected: last 3 entries = rows 7, 57, 107 (ticks 0,1,2 code7); first K-3 entries = 0 pad
    exp = np.zeros(K)
    exp[K - 3:] = [7, 57, 107]
    check(f"K{K} causal content + left-pad", np.allclose(row, exp), f"row={row} exp={exp}")

    # day-start row (tick 0, code 0, day 0): index 0 -> only 1 real entry, K-1 pad
    Xg0 = make_batch(X, np.array([0]), tick_in_day, K)
    r0 = Xg0[0, :, 0]
    exp0 = np.zeros(K)
    exp0[K - 1] = 0.0
    check(f"K{K} day-start only self + pad", np.allclose(r0, exp0), f"r0={r0}")

    # day 1 first tick (code 0): index = n_ticks*n_codes = 150 -> must NOT reference day 0 rows
    Xg1 = make_batch(X, np.array([150]), tick_in_day, K)
    r1 = Xg1[0, :, 0]
    exp1 = np.zeros(K)
    exp1[K - 1] = 150.0
    check(f"K{K} no cross-day chaining", np.allclose(r1, exp1), f"r1={r1}")

print()
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL PASS")
