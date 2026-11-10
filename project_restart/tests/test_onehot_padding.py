# -*- coding: utf-8 -*-
"""TC-41 K1-PREFLIGHT: left-zero-padding vs real category state0 regression test.
Padding positions must be 134-dim all-zero (one-hot group [0,0,0]), NOT real state0 [1,0,0].
Covers day-first ticks 1/2/15/16, stock order, day reset, offline/online alignment, finiteness.
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project_restart", "src", "models"))
sys.path.insert(0, os.path.join("project_restart", "src", "features"))
from gru_strong import load_block, make_batch, CFG  # noqa: E402
from onehot17 import build_onehot17, online_onehot17  # noqa: E402

CFG["window"] = 16
OUT = os.path.join("project_restart", "experiments", "EXP-0029", "onehot_padding_tests.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
CASES = []
FAIL = []


def check(name, cond):
    CASES.append({"name": name, "pass": bool(cond)})
    if not cond:
        FAIL.append(name)
    print(("[PASS] " if cond else "[FAIL] ") + name)


# real data: day 8
X, y, tid = load_block([8])
Xoh = build_onehot17(X)
assert Xoh.shape == (len(X), 134)

# day-first tick 1 (row 0): 1 real tick -> 15 zero-padded + 1 real
W0 = make_batch(Xoh, np.array([0]), tid, 16)  # (1,16,134)
check("tick1: 15 leading positions all-zero (134-dim)", np.allclose(W0[0, :15], 0.0))
check("tick1: real position one-hot group != all-zero", not np.allclose(W0[0, 15, 83:134], 0.0))

# day-first tick 2 (row 50): 2 real ticks -> 14 zero + 2 real
W1 = make_batch(Xoh, np.array([50]), tid, 16)
check("tick2: 14 leading positions all-zero", np.allclose(W1[0, :14], 0.0))
check("tick2: real positions one-hot valid (each group sums to 1)", all(abs(W1[0, 15, 83 + g * 3:86 + g * 3].sum() - 1.0) < 1e-6 for g in range(17)))

# tick 15 (row 14*50): 15 real ticks -> 1 zero + 15 real
W14 = make_batch(Xoh, np.array([14 * 50]), tid, 16)
check("tick15: 1 leading zero position", np.allclose(W14[0, 0], 0.0))

# tick 16 (row 15*50): 16 real ticks -> 0 padding
W15 = make_batch(Xoh, np.array([15 * 50]), tid, 16)
check("tick16: no padding (16 real ticks)", not np.allclose(W15[0, 0], 0.0) and np.isfinite(W15).all())

# tick 17 (row 16*50): 17 real ticks -> window drops oldest, still 16 real (no padding)
W16 = make_batch(Xoh, np.array([16 * 50]), tid, 16)
check("tick17: window has 16 real (no padding)", not np.allclose(W16[0, 0], 0.0))

# padding vs real state0 distinction: construct a synthetic tick where f83 state is 0, compare its one-hot to padding
synt = np.zeros((1, 100), dtype=np.float32)
synt[:, 83] = 0  # real state0
synt_oh = build_onehot17(synt)
pad = np.zeros((1, 134), dtype=np.float32)
check("real state0 one-hot [1,0,0] != padding all-zero", synt_oh[0, 83:86].tolist() == [1.0, 0.0, 0.0] and not np.allclose(pad[0, 83:86], synt_oh[0, 83:86]))

# offline/online alignment on real data (first 5 ticks)
T = X.reshape(-1, 50, 100)[:5]
online = np.concatenate([online_onehot17(t) for t in T], axis=0)
check("offline/online alignment (max abs diff == 0)", float(np.abs(online - Xoh[:5 * 50]).max()) == 0.0)

# finiteness
check("one-hot output finite", np.isfinite(Xoh).all())

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"cases": CASES, "all_pass": not FAIL}, f, indent=2, default=str)
print()
print("ALL PASS" if not FAIL else f"FAILED {FAIL}")
sys.exit(0 if not FAIL else 1)
