# -*- coding: utf-8 -*-
"""TC-40 K0 one-hot encoder unit tests + offline/online alignment."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join("project_restart", "src", "features"))
from onehot17 import build_onehot17, online_onehot17  # noqa: E402

OUT = os.path.join("project_restart", "experiments", "EXP-0028", "onehot_tests_and_alignment.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
CASES = []
FAIL = []


def check(name, cond):
    CASES.append({"name": name, "pass": bool(cond)})
    if not cond:
        FAIL.append(name)
    print(("[PASS] " if cond else "[FAIL] ") + name)


rng = np.random.default_rng(0)
N = 3 * 50
X = np.zeros((N, 100), dtype=np.float32)
X[:, :83] = rng.normal(size=(N, 83)).astype(np.float32)
# category values cycling 0,1,2 across rows for each of 17 cols
for j in range(17):
    X[:, 83 + j] = np.tile([0, 1, 2], 50)[:N]

Xoh = build_onehot17(X)

check("output dim == 134", Xoh.shape == (N, 134))
check("continuous cols preserved (0..82 == X[:,:83])", np.allclose(Xoh[:, :83], X[:, :83]))
check("float32 dtype", Xoh.dtype == np.float32)

# one-hot correctness: row with category value v -> group has 1 at position v
def onehot_of(v):
    return [1.0, 0.0, 0.0] if v == 0 else ([0.0, 1.0, 0.0] if v == 1 else [0.0, 0.0, 1.0])

ok = True
for j in range(17):
    for i in range(N):
        v = int(X[i, 83 + j])
        got = Xoh[i, 83 + j * 3:83 + j * 3 + 3].tolist()
        if got != onehot_of(v):
            ok = False
check("one-hot correct for all 17 cols x 150 rows", ok)

# all three states present & mapped distinctly
check("state 0 -> [1,0,0]", Xoh[0, 83:86].tolist() == [1.0, 0.0, 0.0])
check("state 1 -> [0,1,0] (f84 row1)", Xoh[1, 86:89].tolist() == [0.0, 1.0, 0.0])  # f84 group j=1 at [86,89)
check("state 2 -> [0,0,1] (f84 row2)", Xoh[2, 86:89].tolist() == [0.0, 0.0, 1.0])

# finiteness
check("no NaN/Inf", np.isfinite(Xoh).all())

# stock order: per-row stateless; reorder rows -> corresponding reorder (no cross-row)
perm = np.random.permutation(N)
Xp = build_onehot17(X[perm])
check("per-row stateless (reorder == reorder of outputs)", np.allclose(Xp, Xoh[perm]))

# day reset: stateless (no history) -> same input same output
check("day reset stateless (idempotent)", np.array_equal(build_onehot17(X), Xoh))

# offline/online alignment: build_onehot17 on full == online_onehot17 per tick
T = X.reshape(-1, 50, 100)
online = np.concatenate([online_onehot17(T[t]) for t in range(T.shape[0])], axis=0)
max_abs = float(np.abs(online - Xoh).max())
check("offline/online alignment (max abs diff == 0)", max_abs == 0.0)

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"cases": CASES, "all_pass": not FAIL, "offline_online_max_abs": max_abs}, f, indent=2, default=str)
print()
print("ALL PASS" if not FAIL else f"FAILED {FAIL}")
sys.exit(0 if not FAIL else 1)
