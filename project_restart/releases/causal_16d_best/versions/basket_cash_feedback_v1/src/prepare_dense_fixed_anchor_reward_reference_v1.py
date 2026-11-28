"""Training-fold predictions after stage 21."""
import numpy as np
import experiment_dense_fixed_anchor_reward_bridge_v1 as model
from version_paths import ROOT

OUT = ROOT / "artifacts/dense_fixed_anchor_reward_verified_reference_v1"


def read_p(d):
    if d>=6:
        return np.load(OUT/f"rolling_{d:02d}.npy")
    return model.base.read_p(d)


def rebuild_rolling(final):
    history=[]
    for d in range(6,16):
        if d in (6,8,12):
            bank=model.train(d)
            if history:
                strengths=model.strip.evaluation.scales(history)
        raw,grid,y=[model.onset.settled.load(d,k) for k in ("x","time","y")]
        p=model.base.read_p(d)
        z=model.signals(raw,grid,p,bank)
        q=p.copy()
        if d>=8:
            for h,j in enumerate(final["choices"]):
                if j>=0:
                    q[...,h+1]+=(strengths[j,h]*z[...,j,h]).astype(q.dtype)
        np.save(OUT/f"rolling_{d:02d}.npy",q)
        history.append(np.stack([[model.rel.ref.market.metric(p[...,h+1],y[...,h+1],z[...,j,h])
                                  for h in range(2)] for j in range(3)]))
    np.testing.assert_array_equal(model.strip.evaluation.scales(history),final["strengths"])
