"""Estimate each stock's impact scale separately from the part of its transition profile shared with peers."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch

import prepare_dense_conditional_edge_impact_reference_v1 as best

parent = best.model
direct = parent.direct
base,strip,event,onset,rel = direct.base,direct.strip,direct.event,direct.onset,direct.rel
REFERENCE,read_p = best.OUT,best.read_p
OUT = REFERENCE.parent/'dense_peer_impact_shrinkage_v1'
ARMS = ('stock_scale','peer_shape','stock_and_peer_shape')
C,D = 8,17
SHAPE = (3,50,3,25)


def profiles(days):
    z = sum(parent.moments(d) for d in days).sum(2)
    n,r = z[0]+z[0,...,::-1],z[1]-z[1,...,::-1]
    # Use all other stocks to define the reference response of the market for this name.
    mean = (r.sum(0)-r)/(n.sum(0)-n+1.)
    raw = (r-n*mean)/(n+80)
    weights = n/(n.sum(2,keepdims=True)+1.)
    cov = (weights*mean*raw).sum(2)
    var = (weights*mean**2).sum(2)
    gain = (cov.sum(1)/(var.sum(1)+.01)).clip(-1.,1.)
    scale = gain[:,None,None]*mean
    shape = raw-scale
    # Match names by the shape of the near-term response and how frequently each edge is traversed.
    short = shape[:,0]*np.sqrt(weights[:,0])
    freq = np.sqrt(weights[:,0])
    def standardized(x):
        q = x-x.mean(0)
        return q/max(np.sqrt(np.mean(q*q)),1e-6)
    desc = np.column_stack((standardized(short),standardized(freq)))
    distance = np.mean((desc[:,None]-desc[None,:])**2,axis=2)
    np.fill_diagonal(distance,np.inf)
    near = np.argsort(distance,axis=1)[:,:8]
    w = np.exp(-np.take_along_axis(distance,near,axis=1).clip(0,30))
    w /= w.sum(1,keepdims=True)
    return scale,shape,weights,near,w,gain


def tables(end,report=False):
    a,b = [profiles(range(i,end,2)) for i in (0,1)]
    # Neighbor identities and the payoffs read from those neighbors come from different dates.
    def transfer(left,right):
        return (right[1][left[3]]*left[4][:,:,None,None]).sum(1)
    pa,pb = transfer(b,a),transfer(a,b)
    sa,sb = a[0],b[0]
    scale = np.where(sa*sb>0,np.sign(sa)*np.minimum(abs(sa),abs(sb)),0.)
    peer = .5*(pa+pb)
    ua,ub = a[1]-pa,b[1]-pb
    weight = .5*(a[2]+b[2])
    signal = ((ua*ub*weight).sum(2)).clip(0)
    noise = ((ua-ub)**2*weight).sum(2)/4
    reliability = signal/(signal+noise+.003)
    full = scale+peer+.5*(ua+ub)*reliability[...,None]
    if report:
        return dict(scale_correlation=float(np.corrcoef(a[-1],b[-1])[0,1]),
                    independent_shape_weight=reliability.mean(0).tolist())
    return np.stack((scale,peer,full))


def features(raw,m,p,net,table):
    f,code,vol = direct.fields(raw,m)
    out = np.zeros((*m.shape,D))
    out[:,:,:3] = p
    out[:,:,3:5] = f[:,:,-4:-2]
    out[:,:,5] = vol
    out[:,:,6:8] = p[:,:,1:3]*np.tanh(np.log(vol/.1))[...,None]
    for a in range(0,len(m),64):
        sl = slice(a,min(a+64,len(m)))
        prob,_ = direct.density(f[sl],code[sl],net)
        for k in range(3):
            z = (prob*table[k]).sum(-1)*vol[sl,...,None]*np.array([1,3,8])
            out[sl,:,C+3*k:C+3*k+3] = z
    return strip.center(out)


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y = [onset.settled.load(d,k) for k in ('x','time','y')]
    p = read_p(d)
    tab = tables(d)
    g,n = 0.,0
    for a,b,m,ok in event.sessions(raw,grid):
        t = np.arange(40+d%5,b-a-60,5)
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        valid = gap[t+61]==gap[t]
        x = features(raw[a:b],m,p[a:b],direct.encoder(),tab)[t]
        z = np.concatenate((x,y[a+t,:,1:]),axis=2)
        z -= ((z*valid[...,None]).sum(1)/valid.sum(1)[:,None].clip(1))[:,None]
        rows = z[valid]
        g += rows.T@rows
        n += len(rows)
    return g/n


def train(end):
    co = direct.edge.cycle.ref.fit(np.stack([record(d) for d in range(4,end)]),C,D)
    return np.r_[tables(end).ravel(),direct.encoder(),co.ravel()]


def signals(raw,grid,p,bank):
    cut = int(np.prod(SHAPE))
    tab = bank[:cut].reshape(SHAPE)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x = features(raw[a:b],m,p[a:b],bank[cut:-D*6],tab)
        out[a:b,:,:,1] = x@bank[-D*6:].reshape(D,3,2)[:,:,1]
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
    (OUT/'impact_reliability.json').write_text(json.dumps({end:tables(end,True) for end in (8,12,16)},indent=2))
