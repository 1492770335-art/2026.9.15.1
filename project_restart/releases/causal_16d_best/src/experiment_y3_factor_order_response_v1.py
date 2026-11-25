"""Fit order-flow responses to market-neutral return groups with separate estimated covariances."""
from functools import lru_cache
import sys

import numpy as np
import torch

import experiment_y3_risk_subspace_calibration_v1 as risk
import experiment_y3_independent_order_clocks_v1 as clocks

history,trial,study = risk.history,risk.trial,risk.study
OUT = trial.clock.ART/'y3_factor_order_response_v1'
ARMS = ('one common pricing rule','four strongest non-market factors','intermediate factors',
        'individual stock directions','all three non-market groups','independent group deviations',
        'shuffled asset identities','reliable deviations across dates','individual and intermediate late return',
        'factor-specific excess response beyond pressure')
CANDIDATES = (1,2,3,4,5,7,8,9)
CONTROLS = (0,6)
GROUPS = [(1,5),(5,17),(17,50)]


def features(raw,t):
    sig = clocks.signature.impact.arrival.signals(raw)
    res = []
    for lo,hi in ((0,min(len(raw),2401)),(2401,len(raw))):
        ix = t[(t>=lo)&(t<hi)]-lo
        if not len(ix):
            continue
        v = sig[lo:hi]
        res.append(np.concatenate([v[ix],*(clocks.filters.smooth(v,h)[ix] for h in (4,16,64))],2))
    return np.concatenate((trial.clock.center(np.concatenate(res)),clocks.design_bank(raw,t)),2)


def rotate(z,basis):
    return np.einsum('tsc,sk->tkc',z,basis,optimize=True)


@lru_cache(maxsize=16)
def moments(d):
    raw = trial.clock.book.rawday(d)
    t = np.arange(85+d%11,len(raw)-60,11)
    t = t[(t+60<2401)|(t>=2401)]
    x = features(raw,t)
    y = trial.clock.labels(d).reshape(-1,50,3)
    target = trial.clock.center(np.stack([y[t+h,:,1] for h in (0,20,40)],2)).clip(-4,4)
    grams, grads = [], []
    for basis in risk.geometry():
        v,z = rotate(x,basis),rotate(target,basis)
        g,b = [],[]
        for a,h in GROUPS:
            v1,z1 = v[:,a:h].reshape(-1,v.shape[-1]),z[:,a:h].reshape(-1,3)
            g.append((v1.T@v1)/len(v1))
            b.append((v1.T@z1)/len(v1))
        grams.append(g)
        grads.append(b)
    return np.array(grams),np.array(grads)


@lru_cache(maxsize=7)
def params(end):
    gs,bs = [np.asarray(v,dtype=float) for v in zip(*(moments(d) for d in range(end)))]
    sizes = np.array([b-a for a,b in GROUPS])
    def solve(g,rhs):
        return np.linalg.solve(g+.85*np.diag(g.diagonal().clip(.00002)),rhs)
    ws = []
    for j in range(2):
        g,b = gs[:,j].mean(0),bs[:,j].mean(0)
        pooled = np.average(g,axis=0,weights=sizes)
        base = solve(pooled,np.average(b,axis=0,weights=sizes))
        rows = []
        for i in range(3):
            dates = bs[:,j,i] - gs[:,j,i]@base
            avg = dates.mean(0)
            avg *= abs(avg)/(abs(avg)+dates.std(0)/np.sqrt(end)+1e-6)
            dev = solve(g[i]+.6*pooled,avg)
            stable = solve(g[i]+.6*pooled,avg*abs(np.sign(dates).mean(0)))
            # Regress the order-only columns on all linear pressure channels before fitting.
            basis = np.eye(len(base))[:,32:]
            basis[:32] = -solve(g[i,:32,:32],g[i,:32,32:])
            order = basis@solve(basis.T@g[i]@basis,basis.T@avg)
            rows.append(np.stack((base,base+dev,dev,stable,order),axis=2))
        ws.append(rows)
    return dict(returns=np.array(ws,dtype='float32'),basis=risk.geometry())


def estimates(raw,t,st):
    x = features(raw,t)
    values = []
    for j,basis in enumerate(st['basis']):
        v = rotate(x,basis)
        result = []
        for i,(a,b) in enumerate(GROUPS):
            w = st['returns'][j,i]
            latent = (v[:,a:b]@w.reshape(len(w),-1))
            stock = np.einsum('tkc,sk->tsc',latent,basis[:,a:b],optimize=True)
            result.append(stock.reshape(len(t),50,3,5))
        values.append(np.stack(result,2))
    real,wrong = values
    parts = [real[:,:,:,:,0].sum(2),*[real[:,:,i,:,1] for i in range(3)],real[:,:,:,:,1].sum(2),
             real[:,:,:,:,2].sum(2),wrong[:,:,:,:,1].sum(2),real[:,:,:,:,3].sum(2),
             real[:,:,1:,1:,1].sum(2),real[:,:,:,:,4].sum(2)]
    return trial.clock.center(np.concatenate(parts,2))


def historical_design(d):
    st=params(max(4,d-d%2))
    cache={}
    def build(raw,p,t):
        key=len(raw),t.tobytes()
        if key not in cache:
            cache[key]=estimates(raw,t,st)
        return np.concatenate((trial.clock.center(p[t]),cache[key]),2)
    return build


def fit(rows):
    bounds=np.cumsum([3,3,3,3,3,3,3,3,3,2,3])
    groups=[np.arange(a,b) for a,b in zip(bounds,bounds[1:])]
    return dict(**clocks.linear.gram(rows,[(g,3,.38) for g in groups]),**params(len(rows)+4))


changes=clocks.changes


def correction():
    with np.load(OUT/'model.npz') as archive:
        st=dict(archive)
    def apply(raw,p):
        v=estimates(raw,np.arange(len(raw)),st)
        return changes(np.concatenate((trial.clock.center(p),v),2),st)[int(st['pick'])]
    return apply


def layer():
    return study.layer(sys.modules[__name__],history.incumbent)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
