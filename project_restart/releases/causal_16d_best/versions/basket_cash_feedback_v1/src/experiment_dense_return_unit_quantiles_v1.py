"""Fit identical quantile curves in return units without magnifying residuals of weak early forecast vintages."""
import json
import numpy as np
import torch

import experiment_dense_reflected_quantile_curves_v1 as curves
from prepare_dense_quantile_coverage_reference_v1 import OUT as REFERENCE

onset, rel = curves.onset, curves.rel
OUT = REFERENCE.parent/"dense_return_unit_quantiles_v1"
ARMS = curves.ARMS


def collect():
    OUT.mkdir(exist_ok=True)
    report = []
    for d in range(4,16):
        # The sampled calibration errors were made by this exact chronological parent.
        raw_pred = np.load(curves.REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else onset.settled.head.ranks.band.pairs.net.future.forecast(d)
        p = raw_pred.astype(float)-raw_pred.mean((0,1),keepdims=True,dtype=float)
        y = onset.settled.load(d,"y").astype(float)
        y -= y.mean((0,1),keepdims=True)
        slope = ((p*y).sum((0,1))/(p*p).sum((0,1))).clip(.2,4)
        grid = onset.settled.load(d,"time")
        t = np.concatenate([np.arange(a+30+d%4,b-60,4) for a,b in rel.fields.parent.sessions(grid)])
        target = y[t]-slope*p[t]
        target[...,0] -= target[...,0].mean(1,keepdims=True)
        with np.load(curves.basket.OUT/f"day_{d:02d}.npz") as f:
            old = f["y"]
        np.testing.assert_allclose(target,old*slope,rtol=2e-5,atol=2e-6)
        np.save(OUT/f"return_{d:02d}.npy",target.astype(np.float32))
        report.append(dict(day=d+1,slope=slope.tolist(),label_sd=y.std((0,1)).tolist(),
                           scaled_residual_sd=old.std((0,1),dtype=float).tolist(),return_residual_sd=target.std((0,1)).tolist()))
    (OUT/"residual_units.json").write_text(json.dumps(report,indent=2))


def train(end):
    errors = [np.load(OUT/f"return_{d:02d}.npy") for d in range(4,end)]
    return curves.train(end,errors,OUT)


def design(raw,grid,code,p,bank):
    return curves.design(raw,grid,code,p,bank,OUT)


def predict(raw,grid,p,params):
    z = design(raw,grid,params["code"],p,params["bank"])
    q = p.copy()
    for h in range(3):
        q[...,h] += (rel.relative.features(p,z[...,h])@params["w"][:,h]).astype(q.dtype)
    return q


if __name__=="__main__":
    torch.set_num_threads(1)
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
