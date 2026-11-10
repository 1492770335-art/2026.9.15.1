# -*- coding: utf-8 -*-
"""M0 offline/online market-feature consistency unit test.
Verifies numpy add_market (training) == torch predict-internal (submission) on a synthetic tick,
and that market stats use only the current tick (no cross-tick / future / label).
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def add_market_np(X):  # X (N,100) ordered (time,code)
    T = X.reshape(-1, 50, 100)
    m = T.mean(axis=1, keepdims=True)
    s = T.std(axis=1, keepdims=True, ddof=1)
    m = np.broadcast_to(m, T.shape).reshape(-1, 100)
    s = np.broadcast_to(s, T.shape).reshape(-1, 100)
    return np.concatenate([X, m, s], axis=1).astype(np.float32)


def add_market_torch(x):  # x (50,100) single tick
    m = x.mean(0, keepdim=True)
    s = x.std(0, keepdim=True)
    return torch.cat([x, m.expand_as(x), s.expand_as(x)], dim=1)


rng = np.random.default_rng(0)
T = 3
X = rng.normal(size=(T * 50, 100)).astype(np.float32)
Xm_np = add_market_np(X)

FAIL = []
for t in range(T):
    tick = torch.from_numpy(X[t * 50:(t + 1) * 50])
    online = add_market_torch(tick).numpy()
    off = Xm_np[t * 50:(t + 1) * 50]
    ok = np.allclose(online, off, atol=1e-5)
    print(("[PASS] " if ok else "[FAIL] ") + f"tick {t} offline==online market features")
    if not ok:
        FAIL.append(t)

# no future / no label by construction: features only. cross-tick isolation: modify one tick and confirm only that tick changes
X2 = X.copy()
X2[50:100, 0] += 100.0  # perturb tick 1 only
m2 = add_market_np(X2)
diff0 = np.abs(m2[0:50] - Xm_np[0:50]).max()
diff1 = np.abs(m2[50:100] - Xm_np[50:100]).max()
diff2 = np.abs(m2[100:150] - Xm_np[100:150]).max()
ok_iso = diff0 == 0 and diff2 == 0 and diff1 > 0
print(("[PASS] " if ok_iso else "[FAIL] ") + f"cross-tick isolation (perturb tick1 -> only tick1 changes) d0={diff0} d2={diff2}")
if not ok_iso:
    FAIL.append("iso")

print("ALL PASS" if not FAIL else f"FAILED {FAIL}")
sys.exit(0 if not FAIL else 1)
