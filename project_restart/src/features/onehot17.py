# -*- coding: utf-8 -*-
"""TC-40 K0: discrete category (f83-f99) audit + one-hot encoder (offline & online).
one-hot: replace 17 category cols with 17x3=51 one-hot dims; continuous f0-f82 (83) kept.
Total input 100 -> 134 (83 continuous + 51 one-hot). Stateless, float32, no diff/clip/embedding.
"""
import numpy as np


def build_onehot17(X_raw: np.ndarray) -> np.ndarray:
    """X_raw (N,100) float32 ordered (date,time,code). Returns (N,134) float32.
    layout: [f0..f82 (83) | onehot f83..f99 (17x3=51)]."""
    X_raw = np.asarray(X_raw, dtype=np.float32)
    cont = X_raw[:, :83].astype(np.float32)
    cat = X_raw[:, 83:100].astype(np.int64)  # 17 cols, values in {0,1,2}
    n = len(X_raw)
    onehot = np.zeros((n, 51), dtype=np.float32)
    for j in range(17):
        v = cat[:, j]
        # safety: clip to [0,2]; unknown (if any) maps to all-zero row for that group
        valid = (v >= 0) & (v <= 2)
        rows = np.where(valid)[0]
        onehot[rows, j * 3 + v[rows]] = 1.0
    return np.concatenate([cont, onehot], axis=1).astype(np.float32)


def online_onehot17(x_t: np.ndarray) -> np.ndarray:
    """x_t (50,100) float32 single tick -> (50,134) float32. Stateless, same layout."""
    return build_onehot17(x_t)


def cat_values(x_t: np.ndarray) -> np.ndarray:
    """Return the 17 category values (int) for one tick (50,100)."""
    return x_t[:, 83:100].astype(np.int64)
