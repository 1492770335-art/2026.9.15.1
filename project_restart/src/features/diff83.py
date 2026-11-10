# -*- coding: utf-8 -*-
"""TC-37 D0-B: causal adjacent-tick diff feature constructor (diff83) + online counterpart.
- Offline build_diff83: X_raw (N,100) ordered (date,time,code) -> (N,183) = concat(raw100, diff83).
  diff83 = f(t,stock) - f(t-1,stock) for continuous f0..f82 (cols 0..82); day-first real tick = 0.
  No diff on f83..f99. No cross-stock / cross-day chaining.
- Online Diff83Model state (prev_raw) mirrors this exactly: prev_raw holds the previous tick's
  raw 100-dim features; day reset clears it so day-first tick diff = 0. The K16 window holds 16
  ticks of the 183-dim features; the oldest window tick's diff was computed vs the tick before it
  (the "17th tick"), so it is NOT re-zeroed.
"""
import numpy as np


def build_diff83(X_raw: np.ndarray, dates: np.ndarray) -> np.ndarray:
    """X_raw (N,100) float32, dates (N,) int, rows ordered (date,time,code).
    Returns (N,183) float32 = [raw f0..f99 (100), diff f0..f82 (83)]."""
    X_raw = np.asarray(X_raw, dtype=np.float32)
    n = len(X_raw)
    idx = np.arange(n)
    prev = np.clip(idx - 50, 0, n - 1)
    same = (idx >= 50) & (dates == dates[prev])
    diff = np.zeros((n, 83), dtype=np.float32)
    diff[same] = X_raw[same, :83] - X_raw[prev[same], :83]
    return np.concatenate([X_raw, diff], axis=1).astype(np.float32)


def online_delta(x_t: np.ndarray, prev_raw) -> np.ndarray:
    """Online diff for one tick. x_t (50,100); prev_raw = previous tick raw (50,100) or None.
    Returns (50,83) float32."""
    if prev_raw is None:
        return np.zeros((x_t.shape[0], 83), dtype=np.float32)
    return (x_t[:, :83] - prev_raw[:, :83]).astype(np.float32)


def online_feat183(x_t: np.ndarray, prev_raw) -> np.ndarray:
    """Online 183-dim feature for one tick = concat(raw, online_delta)."""
    return np.concatenate([x_t, online_delta(x_t, prev_raw)], axis=1).astype(np.float32)
