"""Use the issued original forecasts to learn a frequency-independent rotation across stock baskets."""
import json
import sys
from pathlib import Path

import numpy as np

import version_paths as base
from version_paths import released

OUT = base.ROOT / "artifacts/dense_release_spectral_readout_v1"
TAPE = base.TAPE
ARMS = ("local", "basket", "mixed")


def bands(p):
    scaled = p.astype(float) * base.UNITS
    return np.diff(scaled, prepend=0., axis=2)


def center(x):
    return x - x.mean(1, keepdims=True)


def record(p, y):
    x = center(bands(p))
    a, b = p[..., 2].astype(float), y[..., 2].astype(float)
    a -= a.mean()
    b -= b.mean()
    beta = np.clip((a*b).mean()/(a*a).mean(), .35, 3.)
    err = center(b/beta-a)
    # A common per-stock Gram would discard the covariance between different names.
    g = np.einsum("tsj,tuk->jksu", x, x, optimize=True) / len(x)
    cross = np.stack([x[..., j].T @ err / len(x) for j in range(3)])
    return g, cross, (a*a).mean()


def weights(history, arm):
    g, cross, var = [np.stack(rows) for rows in zip(*history)]
    mean_g = g.mean(0)
    if arm == "local":
        axes = np.eye(50)
    else:
        # Give each forecast interval equal mass after rescaling its forecast variance.
        geometry = sum(mean_g[j,j]/np.trace(mean_g[j,j]) for j in range(3))
        _, axes = np.linalg.eigh(geometry)
        axes = axes[:, 1:]  # Constant market direction is removed from x and err.
    local_g = np.einsum("si,jksu,ui->ijk", axes, mean_g, axes, optimize=True)
    slopes = np.einsum("si,djsu,ui->dij", axes, cross, axes, optimize=True)
    grad = slopes.mean(0)
    # Only model differences in reliability between individual names or baskets.
    pooled = np.linalg.solve(local_g.sum(0)+np.eye(3)*1e-8, grad.sum(0))
    grad -= local_g @ pooled
    stderr = slopes.std(0, ddof=1) / np.sqrt(len(history))
    grad *= abs(grad)/(abs(grad)+stderr+1e-8)
    diag = local_g.diagonal(axis1=1, axis2=2)
    reg = .7*diag + .8*diag.mean(0)
    w = np.linalg.solve(local_g + np.eye(3)[None]*reg[:,:,None], grad[..., None])[...,0]
    exposure = np.einsum("ij,ijk,ik->", w, local_g, w)/50.
    w *= min(1., .08*np.sqrt(var.mean()/max(exposure, 1e-9)))
    return np.stack([(axes*w[:,j]) @ axes.T for j in range(3)])


def update(p, w):
    x = center(bands(p))
    return center(sum(x[...,j] @ w[j] for j in range(3))).astype("float32")


def metric(p, y, z):
    flat = np.stack([p.ravel(), y.ravel(), z.ravel()]).astype(float)
    return np.pad(flat@flat.T, ((1,0),(1,0))) + np.block([[np.array([[flat.shape[1]]]),flat.sum(1)[None]], [flat.sum(1)[:,None],np.zeros((3,3))]])


def gain(g):
    cov = g[1:,1:]/g[0,0]-np.outer(g[0,1:],g[0,1:])/g[0,0]**2
    p,y,z=cov.diagonal()
    return float((cov[0,1]+cov[1,2])/np.sqrt((p+2*cov[0,2]+z)*y)-cov[0,1]/np.sqrt(p*y))


def rolling():
    OUT.mkdir(exist_ok=True)
    hist, evaluated = [], []
    for day in range(4,16):
        p = np.load(TAPE/f"forecast_{day:02d}.npy")
        y = np.load(base.CACHE/f"day_{day:02d}_y.npy",mmap_mode="r")
        if day in (8,12):
            w = [weights(hist, arm) for arm in ARMS[:2]]
            w.append((w[0]+w[1])*.5)
        if day >= 8:
            rows=[]
            for coef in w:
                z = update(p,coef)
                np.testing.assert_array_equal(z[:157],update(p[:157],coef))
                groups = [metric(p[i::5,:,2],y[i::5,:,2],z[i::5]) for i in range(5)]
                wrong = metric(p[...,2],y[...,2],np.roll(z,13,axis=1))
                rows.append((groups,wrong))
            evaluated.append(rows)
        hist.append(record(p,y))
    report={}
    for i,arm in enumerate(ARMS):
        scores=[]
        for fold in (evaluated[:4],evaluated[4:]):
            phase=np.sum([r[i][0] for r in fold],axis=0)
            daily=[gain(sum(r[i][0])) for r in fold]
            scores.append(dict(full=gain(phase.sum(0)),phases=[gain(v) for v in phase],daily=daily,
                               wrong=gain(sum(r[i][1] for r in fold))))
        report[arm]=scores
    eligible=[arm for arm, folds in report.items() if all(f["full"]>max(0,f["wrong"]) and min(f["phases"])>0 for f in folds)
              and sum(x>0 for f in folds for x in f["daily"])>=5]
    best=max(eligible,key=lambda arm:min(f["full"] for f in report[arm])) if eligible else None
    if best:
        coeff=(weights(hist,"local")+weights(hist,"basket"))/2 if best=="mixed" else weights(hist,best)
        np.save(OUT/"weights.npy",coeff)
    (OUT/"rolling.json").write_text(json.dumps(dict(selected=best,folds=report),indent=2))
    print("Original Y3 stock readout:",best, {arm:[round(f["full"],7) for f in folds] for arm,folds in report.items()},flush=True)


def outer():
    result=json.loads((OUT/"rolling.json").read_text())
    if not result["selected"]:
        return
    rows=[]
    for day in range(16,20):
        raw=np.load(base.CACHE/f"day_{day:02d}_x.npy",mmap_mode="r")
        p=released(day,raw)
        y=np.load(base.CACHE/f"day_{day:02d}_y.npy",mmap_mode="r")
        q=p.copy()
        q[...,2]+=update(p,np.load(OUT/"weights.npy"))
        rows.append((p,y,q))
    pooled=base.score(*(np.concatenate(col) for col in zip(*rows)))
    assert pooled["rows"] == 936250 and abs(pooled["base_score"]-.37095867568143737)<1e-8
    (OUT/"outer.json").write_text(json.dumps(dict(pooled=pooled,daily=[base.score(*r) for r in rows]),indent=2))
    print(pooled,flush=True)


if __name__=="__main__":
    rolling()
    outer()
