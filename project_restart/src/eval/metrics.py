# -*- coding: utf-8 -*-
"""Pooled Pearson + official score rounding, mirroring script/evaluate.py L61-L67."""
import numpy as np


def pooled_pearson_raw(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """p, y: (N, 3) float64. Returns unrounded corr per target.
    Matches evaluate.py: moments accumulated over ALL evaluated rows (days pooled).
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(p)
    pp = (p * p).sum(0)
    yy = (y * y).sum(0)
    py = (p * y).sum(0)
    ps = p.sum(0)
    ys = y.sum(0)
    denom = np.sqrt(np.maximum(pp - ps * ps / n, 0) * np.maximum(yy - ys * ys / n, 0))
    if (denom == 0).any():
        raise ValueError("invalid predict")
    return (py - ps * ys / n) / denom


def official_score(p: np.ndarray, y: np.ndarray) -> dict:
    """Mirror script/evaluate.py L67 exactly: score = round(unrounded_corr @ weights, 4);
    corr0/1/2 are individually rounded for display only."""
    corr = pooled_pearson_raw(p, y)
    raw = float(corr @ np.array([0.2, 0.3, 0.5]))
    return {
        "corr0": float(round(float(corr[0]), 4)),
        "corr1": float(round(float(corr[1]), 4)),
        "corr2": float(round(float(corr[2]), 4)),
        "raw_score": raw,
        "score": round(raw, 4),
    }
