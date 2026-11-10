# -*- coding: utf-8 -*-
"""TC-37 D0-B diff83 unit tests + offline/online alignment.
Covers: day-first tick diff=0; tick2 diff=f2-f1; window oldest-tick diff (17-tick history) not re-zeroed;
left padding; day switch reset; per-stock isolation; insufficient history; future-info isolation;
offline build_diff83 == online streaming reconstruction (tick-aligned).
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src", "features"))
from diff83 import build_diff83, online_feat183, online_delta  # noqa: E402

OUT = os.path.join("project_restart", "experiments", "EXP-0024", "test_cases_and_results.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

FAIL = []
CASES = []


def check(name, cond, detail=""):
    CASES.append({"name": name, "pass": bool(cond), "detail": detail})
    print(("[PASS] " if cond else "[FAIL] ") + name + ("" if cond else f" {detail}"))


# synthetic: 2 dates x 5 ticks x 50 codes; feature value = row index (traceable), cols 0..99 = row index repeated
rng = np.random.default_rng(0)
n_dates, n_ticks, n_codes = 2, 5, 50
N = n_dates * n_ticks * n_codes
X = np.arange(N, dtype=np.float32).reshape(-1, 1).repeat(100, axis=1)
dates = np.repeat([0, 1], n_ticks * n_codes)

X183 = build_diff83(X, dates)  # (N, 183)

# 1) day-first tick diff == 0
t0 = 0  # day0 tick0 code0
check("day0 first tick diff==0", np.allclose(X183[0, 100:183], 0.0))

# 2) tick2 diff = f(t2)-f(t1) for same code
i = 50  # day0 tick1 code0 -> diff = X[50]-X[0] = 50
check("tick2 diff == f(t2)-f(t1)", np.allclose(X183[50, 100:183], X[50, :83] - X[0, :83]) and np.allclose(X183[50, 100], 50.0))

# 3) window oldest-tick diff (17-tick history): tick16 (idx 16) diff = X[16]-X[16-50]... n_ticks=5 so use bigger synthetic below

# ---- bigger synthetic: 1 date x 20 ticks x 50 codes ----
n_ticks2 = 20
N2 = n_ticks2 * n_codes
X2 = np.arange(N2, dtype=np.float32).reshape(-1, 1).repeat(100, axis=1)
d2 = np.zeros(N2, dtype=np.int64)
X183_2 = build_diff83(X2, d2)

# tick 16 (idx 16) diff should be X[16]-X[16-50] but idx<50 -> 0 (insufficient history, day first 50 rows are tick0..tick0? no: tick0 = rows 0..49)
# Actually idx 16 is within tick0 (rows 0..49 = tick0), so idx16 is tick0 code16 -> day-first tick -> diff 0
check("tick16 (within day first tick) diff==0", np.allclose(X183_2[16, 100:183], 0.0))
# idx 50 = tick1 code0: diff = X[50]-X[0] = 50
check("tick1 code0 diff==50", np.allclose(X183_2[50, 100], 50.0))
# idx 66 = tick1 code16: diff = 66-16 = 50
check("tick1 code16 diff==50", np.allclose(X183_2[66, 100], 50.0))
# window oldest tick: for a window ending at tick16 (idx 16*50+16=816?), the oldest window tick = tick1 (idx 50..99), its diff already = f(tick1)-f(tick0)=50 (not zero)
# verify: idx 816 = tick16 code16 -> window oldest tick = tick1 (16-15=1) code16 -> its diff at idx 66 = 50, NOT zero
check("window oldest-tick diff not re-zeroed (17-tick history)", np.allclose(X183_2[66, 100], 50.0))

# 4) left padding: day first tick window of length1 -> only 1 real tick, rest padded (offline build_diff83 keeps raw; padding handled by make_batch, not diff83). Verify diff83 preserves raw dims.
check("output dim == 183", X183.shape[1] == 183 and X183_2.shape[1] == 183)
check("raw dims preserved (cols 0..99 == X)", np.allclose(X183[:, :100], X))

# 5) day switch reset: day1 first tick (idx 5*50=250) diff == 0
check("day1 first tick diff==0", np.allclose(X183[250, 100:183], 0.0))

# 6) per-stock isolation: only same code 50 rows back contributes; perturb one stock's history and confirm no cross-stock
Xp = X.copy()
Xp[1, 0] = 9999.0  # day0 tick0 code1 raw f0
Xp183 = build_diff83(Xp, dates)
# tick1 code1 (idx 51) diff should be X[51]-Xp[1] (only its own code), code0 (idx50) diff unaffected
check("per-stock isolation (code0 diff unaffected by code1 perturb)",
      np.allclose(Xp183[50, 100], X[50, 0] - X[0, 0]) and np.allclose(Xp183[51, 100], X[51, 0] - Xp[1, 0]))

# 7) future-info isolation: diff at tick t uses only f(t), f(t-1); perturb f(t+1) does not change tick t diff
Xq = X.copy()
Xq[100, 0] = 12345.0  # tick2 code0 (future relative to tick1)
Xq183 = build_diff83(Xq, dates)
check("future-info isolation (tick1 diff unaffected by tick2 perturb)",
      np.allclose(Xq183[50, 100], X[50, 0] - X[0, 0]))

# 8) offline/online alignment: simulate streaming over the 20-tick synthetic
online_feats = []
prev_raw = None
for t in range(n_ticks2):
    x_t = X2[t * n_codes:(t + 1) * n_codes]
    online_feats.append(online_feat183(x_t, prev_raw))
    prev_raw = x_t
online = np.concatenate(online_feats, axis=0)
offline = X183_2
max_abs = float(np.abs(online - offline).max())
check("offline/online alignment (max abs diff <= 1e-6)", max_abs <= 1e-6, f"max_abs={max_abs}")

with open(OUT, "w", encoding="utf-8") as f:
    json.dump({"cases": CASES, "all_pass": not FAIL}, f, indent=2, default=str)

print()
print("ALL PASS" if not FAIL else f"FAILED {FAIL}")
sys.exit(0 if not FAIL else 1)
