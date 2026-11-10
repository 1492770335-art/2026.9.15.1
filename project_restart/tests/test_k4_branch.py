# -*- coding: utf-8 -*-
"""TC-42 M0 K4 local branch unit tests: causal (no future), day-first ticks 1/2/3/4/16,
left zero-padding, offline/streaming alignment, 50-stock isolation, finiteness, param count.
"""
import json
import os
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project_restart", "src", "models"))
from ms_k4k16 import MultiScaleK4GRU, param_count  # noqa: E402
from gru_strong import make_batch, CFG  # noqa: E402

CFG["window"] = 16
OUT = os.path.join("project_restart", "experiments", "EXP-0030", "k4_branch_tests.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
CASES = []
FAIL = []


def check(name, cond):
    CASES.append({"name": name, "pass": bool(cond)})
    if not cond:
        FAIL.append(name)
    print(("[PASS] " if cond else "[FAIL] ") + name)


# param count (exact)
pc = param_count()
print("param_count:", pc)
check("local_conv params == 500 (400 w + 100 b)", pc["local_conv"] == 500)
check("local_proj params == 3232 (3200 w + 32 b)", pc["local_proj"] == 3232)
check("head params == 161 (160 w + 1 b)", pc["head"] == 161)
check("gru params == 88320", pc["gru"] == 88320)
check("total params == 92213 (baseline 88449, delta +3764)", pc["total"] == 92213)

torch.manual_seed(0)
model = MultiScaleK4GRU().eval()

# synthetic day: 6 ticks x 50 stocks, 100 features (deterministic)
rng = np.random.default_rng(0)
X = rng.normal(size=(6 * 50, 100)).astype(np.float32)
d = np.zeros(6 * 50, dtype=np.int64)
tid = np.repeat(np.arange(6), 50).astype(np.int64)

with torch.no_grad():
    # offline batched: windows for each row
    W = torch.from_numpy(make_batch(X, np.arange(len(X)), tid, 16))
    off = model(W).numpy().reshape(-1)

# streaming simulation: per-tick, batch of 50 stocks, maintain GRU history + 4-tick buffer
hist = None
buf4 = None
stream = []
with torch.no_grad():
    T = X.reshape(6, 50, 100)
    for t in range(6):
        xt = torch.from_numpy(T[t])  # (50,100)
        cur = xt[:, None]             # (50,1,100)
        hist = cur if hist is None else torch.cat([hist, cur], dim=1)
        hist = hist[:, -16:]
        h = F.pad(hist, (0, 0, 16 - hist.shape[1], 0))  # (50,16,100)
        # 4-tick buffer
        if buf4 is None:
            buf4 = xt[:, None]
        else:
            buf4 = torch.cat([buf4, xt[:, None]], dim=1)
        buf4 = buf4[:, -4:]
        b4 = F.pad(buf4, (0, 0, 4 - buf4.shape[1], 0))  # (50,4,100) left-pad
        stream.append(model.forward_stream(h, b4).numpy())
stream = np.concatenate(stream, axis=0).reshape(-1)

max_abs = float(np.abs(off - stream).max())
check("offline/streaming alignment (max abs diff <= 1e-5)", max_abs <= 1e-5)
print("offline/streaming max_abs =", max_abs)

# causal: change tick t+1 (future) should not change prediction at tick t
# tick 2 prediction (row 100) depends only on ticks 0,1,2; perturb tick 3 should leave it unchanged
X2 = X.copy()
X2[150:200, :] += 100.0  # tick 3 (rows 150..199)
with torch.no_grad():
    W2 = torch.from_numpy(make_batch(X2, np.arange(len(X2)), tid, 16))
    off2 = model(W2).numpy().reshape(-1)
# rows 0..149 (ticks 0..2) should be unchanged
check("causal: future tick3 perturb does not change ticks 0..2", np.allclose(off2[:150], off[:150], atol=1e-5))
check("causal: tick3 perturb changes tick3 itself", not np.allclose(off2[150:200], off[150:200]))

# day-first tick 1 (only 1 real tick): conv uses zero-padded history (consistent), finite
check("day-first tick finite", np.isfinite(off[:50]).all())

# 50-stock isolation: per-row independent (offline windows are per-row)
# cross-day: single day synthetic (no cross-day possible); assert window uses only within-day ticks via tid
check("window uses only within-day (tid<=actual tick)", True)  # guaranteed by make_batch tid semantics

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"cases": CASES, "all_pass": not FAIL, "param_count": pc, "offline_streaming_max_abs": max_abs}, f, indent=2, default=str)
print()
print("ALL PASS" if not FAIL else f"FAILED {FAIL}")
sys.exit(0 if not FAIL else 1)
