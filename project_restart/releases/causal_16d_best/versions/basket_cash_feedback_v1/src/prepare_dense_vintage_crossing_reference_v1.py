"""Training-fold predictions after stage 22."""
import numpy as np
import experiment_dense_vintage_crossing_shrinkage_v1 as model
from version_paths import ROOT

OUT = ROOT / "artifacts/dense_vintage_crossing_verified_reference_v1"


def read_p(d):
    if d<6:
        return model.read_p(d)
    return np.load(OUT/f'rolling_{d:02d}.npy')


def rebuild_rolling(params,model=model,directory=OUT):
    history=[]
    for d in range(6,16):
        if d in (6,8,12):
            bank=model.train(d)
            if history:
                strength=model.strip.evaluation.scales(history)
        p=model.read_p(d)
        raw,grid,y=[model.onset.settled.load(d,k) for k in ('x','time','y')]
        z=model.signals(raw,grid,p,bank)
        q=p.copy()
        if d>=8:
            for h,j in enumerate(params['choices']):
                if j>=0:
                    q[...,h+1]+=(strength[j,h]*z[...,j,h]).astype(q.dtype)
        np.save(directory/f'rolling_{d:02d}.npy',q)
        history.append(np.stack([[model.rel.ref.market.metric(p[...,h+1],y[...,h+1],z[...,j,h])
                                  for h in range(2)] for j in range(3)]))
    np.testing.assert_array_equal(model.strip.evaluation.scales(history),params['strengths'])
