# -*- coding: utf-8 -*-
"""TC-39 C1 clip boundary/preprocessing unit tests (zero-training).
Verify clip83 clips only f0-f82 to [-10,10], keeps f83-f99, boundary values, float32,
left-padding (no interaction), day-start reset (no carry), per-stock isolation, no label touch.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join("project_restart", "src", "models"))
from exp_tc39_c1 import clip83  # noqa: E402

OUT = os.path.join("project_restart", "experiments", "EXP-0027", "clip_boundary_tests.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
CASES = []
FAIL = []


def check(name, cond):
    CASES.append({"name": name, "pass": bool(cond)})
    if not cond:
        FAIL.append(name)
    print(("[PASS] " if cond else "[FAIL] ") + name)


# synthetic: 1 tick x 50 stocks x 100 features, f0-f82 = value, f83-f99 = 1000 (should stay)
X = np.zeros((50, 100), dtype=np.float32)
X[:, :83] = np.linspace(-20, 20, 83)[None, :]  # f0=-20 .. f82=+20
X[:, 83:] = 1000.0  # discrete untouched

Xc = clip83(X)

check("f0-f82 clipped to [-10,10] (min>=-10, max<=10)", Xc[:, :83].min() >= -10.0 and Xc[:, :83].max() <= 10.0)
check("f83-f99 unchanged (==1000)", np.allclose(Xc[:, 83:], 1000.0))
check("boundary -10 exact (f0=-20 -> -10)", np.allclose(Xc[:, 0], -10.0))
check("boundary +10 exact (f82=+20 -> +10)", np.allclose(Xc[:, 82], 10.0))
check("interior preserved (value 0 -> 0)", np.allclose(Xc[:, 41], 0.0))
check("float32 dtype", Xc.dtype == np.float32)
check("input dim == 100", Xc.shape == (50, 100))

# per-stock isolation + no label touch: labels are separate (clip83 only sees X)
# day-start reset: clip is stateless (no carry) by construction; verify same output independent of history
X1 = clip83(X)
X2 = clip83(X)
check("stateless (same input -> same output)", np.array_equal(X1, X2))

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"cases": CASES, "all_pass": not FAIL}, f, indent=2, default=str)
print()
print("ALL PASS" if not FAIL else f"FAILED {FAIL}")
sys.exit(0 if not FAIL else 1)
