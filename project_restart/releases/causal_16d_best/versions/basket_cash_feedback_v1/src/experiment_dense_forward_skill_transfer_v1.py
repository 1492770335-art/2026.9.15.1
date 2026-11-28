"""Learn whether already-resolved near and distal marginal payoffs predict the next block's Y3 skill."""
from functools import lru_cache
import sys

import numpy as np
import torch

import prepare_dense_forecast_overlap_reference_v1 as best

parent = best.model
pre,strip,event,onset,rel = parent.pre,parent.strip,parent.event,parent.onset,parent.rel
REFERENCE,read_p = best.OUT,best.read_p
OUT = REFERENCE.parent/'dense_forward_skill_transfer_v1'
ARMS = ('recent_near_legs','recent_distal_leg','cross_horizon_confirmation')
K, F = 9, 9
LAGS = ((0,5),(5,20),(20,60))


def hedge(x, ok):
    controls = np.concatenate((np.ones((*x.shape[:2],1)),x[:,:,:3]),axis=2)*ok[...,None]
    xt = controls.swapaxes(1,2)
    beta = np.linalg.pinv(xt@controls,hermitian=True,rcond=1e-10)@(xt@x[:,:,3:])
    return (x[:,:,3:]-controls@beta)*ok[...,None]


def payout(z, path, gap, t, lo, hi):
    ok = gap[t+hi+1]==gap[t]
    r = path[t+hi]-path[t+lo]
    r -= (r*ok).sum(1,keepdims=True)/ok.sum(1,keepdims=True).clip(1)
    return ((r*ok).reshape(-1)@z[t].reshape(-1,K))/max(1,ok.sum())


def blocks(raw,m,ok,p,bank,model):
    x = hedge(parent.features(raw,p,ok,bank,model),ok)
    z = np.zeros_like(x)
    path = m.cumsum(0)
    gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
    # Normalization uses past raw features. Payoff histories use past, fully matured windows.
    seed = np.array([.3,.2,.2,.04,.05,.05,.02,.02,.02])**2
    sq = seed.copy()
    n = 1.
    histories = [[],[],[]]
    cursors = np.zeros(3,int)
    gram = np.eye(K)
    for start in range(0,len(m),96):
        stop = min(start+96,len(m))
        if start:
            vx = x[start-96:start:4].reshape(-1,K)
            sq,n = .92*sq+(vx**2).mean(0), .92*n+1
            prior_z = z[start-96:start:4].reshape(-1,K)
            gram = .85*gram+.15*(prior_z.T@prior_z/len(prior_z))
        z[start:stop] = np.clip(x[start:stop]/np.sqrt((sq/n).clip(1e-8)),-8.,8.)
        for j,(lo,hi) in enumerate(LAGS):
            t = np.arange(cursors[j],start-hi+1,4)
            if len(t):
                cursors[j] = t[-1]+4
                histories[j].append(payout(z,path,gap,t,lo,hi)/np.sqrt(hi-lo))
        if min(map(len,histories))<3:
            continue
        feats = []
        for hist in histories:
            rows = np.stack(hist[-12:])
            age = .85**np.arange(len(rows)-1,-1,-1)
            avg = age@rows/age.sum()
            older = np.mean(rows[:-2],axis=0)
            recent = rows[-2:].mean(0)
            confirmed = np.where(older*recent>0,np.sign(older)*np.minimum(abs(older),abs(recent)),0.)
            feats.extend((avg,recent-older,confirmed))
        yield start,stop,z,np.stack(feats,axis=1),gram,path,gap


@lru_cache(maxsize=12)
def record(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    p = read_p(day)
    base = pre.encoder()
    net = pre.restore(base)
    mats = []
    for a,b,m,ok in event.sessions(raw,grid):
        for start,stop,z,f,_,path,gap in blocks(raw[a:b],m,ok,p[a:b],base,net):
            t = np.arange(start,min(stop,len(m)-60),4)
            if len(t)>=10:
                y = payout(z,path,gap,t,0,60)
                v = np.column_stack((f,y))
                mats.append(v.T@v/K)
    return np.mean(mats,axis=0)


def train(end):
    gs = np.stack([record(d) for d in range(4,end)])
    g = gs.mean(0)
    w = np.zeros((F,3))
    for j,idx in enumerate((np.arange(6),np.arange(6,9),np.arange(F))):
        r = gs[:,idx,-1]
        avg = r.mean(0)
        safe = avg**2/(avg**2+2*r.var(0)/len(gs)+1e-12)
        w[idx,j] = pre.peer.base.solve(g[np.ix_(idx,idx)],avg*safe,.85)
    cash = np.mean([parent.parent.hedged.teacher.live.calibration(d) for d in range(4,end)],0)
    unit = np.clip(cash[1,0]/cash[1,1],.03,6.)
    return np.r_[unit,w.ravel(),pre.encoder()]


def signals(raw,grid,p,params):
    weights = params[1:1+F*3].reshape(F,3)
    bank = params[1+F*3:]
    net = pre.restore(bank)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        for start,stop,z,f,g,_,_ in blocks(raw[a:b],m,ok,p[a:b],bank,net):
            w = np.linalg.solve(g+.9*np.eye(K),f@weights)/params[0]
            out[a+start:a+stop,:,:,1] = strip.center(z[start:stop]@w)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
