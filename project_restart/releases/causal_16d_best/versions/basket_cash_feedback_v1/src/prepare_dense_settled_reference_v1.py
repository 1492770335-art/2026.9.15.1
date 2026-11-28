"""Training-fold predictions after stage 1."""
import numpy as np
import experiment_dense_value_maturity_updates_v1 as model
from version_paths import ROOT

OUT = ROOT / "artifacts/dense_settled_verified_reference_v1"

rel = model.rel


def update(module, raw, grid, p, code, bank, weights):
    z = module.views(raw,grid,code,bank,p)
    delta = (rel.relative.features(p,z)@weights).astype(p.dtype)
    q = p.copy()
    q[...,2] += delta
    return q
