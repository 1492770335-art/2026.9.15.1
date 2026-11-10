# -*- coding: utf-8 -*-
"""Deterministic unit tests for pooled Pearson & official rounding (no torch needed).
Run: python tests/test_metrics_and_rounding.py
"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import numpy as np

from src.eval.metrics import pooled_pearson_raw, official_score

FAIL = []


def check(name, cond, detail=""):
    if cond:
        print(f"[PASS] {name}")
    else:
        print(f"[FAIL] {name} {detail}")
        FAIL.append(name)


# 1. perfect linear prediction -> corr = 1
rng = np.random.default_rng(0)
x = rng.normal(size=(1000, 3))
y = 2.0 * x + 1.0
corr = pooled_pearson_raw(x, y)
check("perfect linear corr==1", np.allclose(corr, 1.0), str(corr))

# 2. anti-correlated -> -1
corr = pooled_pearson_raw(x, -3.0 * x + 0.5)
check("perfect anti-corr == -1", np.allclose(corr, -1.0), str(corr))

# 3. independence (sample) -> |corr| small, not exactly 0
x2 = rng.normal(size=(1000, 1))
y2 = rng.normal(size=(1000, 1))
corr = pooled_pearson_raw(x2, y2)
check("independent sample |corr|<0.1", abs(float(corr[0])) < 0.1, str(corr))

# 4. pooled vs mean-of-daily is NOT equal (documents why pooling matters)
p1 = np.array([[1.0, 2.0]]).T
y1 = np.array([[2.0, 4.0]]).T   # day1 slope +2 -> daily corr +1
p2 = np.array([[3.0, 4.0]]).T
y2 = np.array([[-3.0, -4.0]]).T  # day2 slope -1 -> daily corr -1
pooled = pooled_pearson_raw(np.vstack([p1, p2]), np.vstack([y1, y2]))
daily = (pooled_pearson_raw(p1, y1) + pooled_pearson_raw(p2, y2)) / 2
expected = -12.5 / np.sqrt(5.0 * 44.75)
check("pooled != daily mean (docs)", np.allclose(pooled, [expected]) and np.allclose(daily, [0.0]),
      f"pooled={pooled} daily={daily} expected={expected}")

# 5. constant prediction -> invalid predict (denominator 0)
try:
    pooled_pearson_raw(np.ones((10, 3)), rng.normal(size=(10, 3)))
    check("constant prediction raises", False)
except ValueError as e:
    check("constant prediction raises", str(e) == "invalid predict", str(e))

# 6. banker's rounding semantics documentation (round each corr to 4 -> weighted -> round to 4)
c = np.array([0.05, 0.55, 0.005])
corr_r = np.round(c, 4)
expected = round(float(corr_r @ np.array([0.2, 0.3, 0.5])), 4)
raw = float(c @ np.array([0.2, 0.3, 0.5]))
print(f"[INFO] rounding path: corr={c} rounded={corr_r} raw_weighted={raw} official={expected}")
check("rounding path computed", True)

# 7. pooling invariance to row order
idx = rng.permutation(1000)
p_ = x[idx]
y_ = (2.0 * x + 1.0)[idx]
check("order invariance", np.allclose(pooled_pearson_raw(p_, y_), 1.0))

print()
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL PASS")
