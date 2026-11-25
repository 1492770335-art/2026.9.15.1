"""Share order-response parameters smoothly across the risk spectrum of the fifty-stock panel."""
from functools import lru_cache
import sys
import numpy as np
import torch

import y3_factor_order_references as history

parent,trial,study=history.incumbent,history.trial,history.study
OUT=trial.clock.ART/'y3_continuous_factor_order_v1'
ARMS=('hard groups from first four days','hard groups with updated covariance','overlapping risk groups',
      'continuous eigenvalue response','stable covariance across dates','uncertain factors pooled',
      'permuted spectrum','only individual directions','soft minus hard response',
      'share later returns across nearby modes')
CANDIDATES=(1,2,3,4,5,7,8,9)
CONTROLS=(0,6)


@lru_cache(maxsize=16)
def cov(d):
    y=trial.clock.labels(d).reshape(-1,50,3)
    t=np.arange(65+d%7,len(y)-60,7)
    t=t[(t+60<2401)|(t>=2401)]
    r=trial.clock.center(y[t,:,1].astype(float))
    return np.corrcoef(r.T)


@lru_cache(maxsize=7)
def geometries(end):
    base=parent.risk.geometry()[0][:,1:].astype(float)
    gs=np.stack([base.T@cov(d)@base for d in range(end)])
    def directions(g):
        s,u=np.linalg.eigh(g)
        u=base@u[:,::-1]
        return s[::-1],u
    s,u=directions(gs.mean(0))
    a,b=[gs[i::2].mean(0) for i in (0,1)]
    _,rep=directions(.5*(a@b+b@a))
    hard=np.eye(3)[np.repeat(np.arange(3),[4,12,33])]
    pos=np.arange(49)
    def blend(x,centers,widths):
        logp=-.5*((x[:,None]-centers)/widths)**2-np.log(widths)
        weight=np.exp(logp-logp.max(1,keepdims=True))
        return weight/weight.sum(1,keepdims=True)
    soft=blend(np.log1p(pos),np.log1p([1.5,9,30]),np.array([.55,.60,.65]))
    value=blend(np.log(s.clip(.01)),np.log(s[[1,9,30]].clip(.01)),np.array([.32,.29,.45]))
    # For unstable eigenvectors, estimate exposure to three groups on two disjoint date samples.
    prob=np.mean([(u.T@directions(x)[1])**2@hard for x in (a,b)],0)
    spread=np.stack([base,u,u,u,rep,u,np.roll(u,17,axis=0)])
    mix=np.stack([hard,hard,soft,value,soft,prob,soft])
    return spread.astype('float32'),mix.astype('float32')


@lru_cache(maxsize=16)
def moment(d):
    raw=trial.clock.book.rawday(d)
    t=np.arange(90+d%11,len(raw)-60,11)
    t=t[(t+60<2401)|(t>=2401)]
    # Preserve the full 9 ordered clocks so this comparison changes factor sharing only.
    x=parent.features(raw,t)
    y=trial.clock.labels(d).reshape(-1,50,3)
    z=trial.clock.center(np.stack([y[t+h,:,1] for h in (0,20,40)],2)).clip(-4,4)
    inp=np.concatenate((x,z),2)
    bases,mixes=geometries(max(4,d-d%2))
    seen={}
    rows=[]
    for u,mix in zip(bases,mixes):
        key=u.tobytes()
        if key not in seen:
            rv=parent.rotate(inp,u).transpose(1,0,2).copy()
            seen[key]=rv.transpose(0,2,1)@rv/len(t)
        rows.append((mix.T@seen[key].reshape(49,-1)/mix.sum(0)[:,None]).reshape(3,inp.shape[2],-1))
    return np.array(rows)


@lru_cache(maxsize=7)
def state(end):
    records=np.asarray([moment(d) for d in range(end)],dtype=float)
    width=records.shape[-1]-3
    pool=np.average(records.mean(0)[0],axis=0,weights=[4,12,33])
    def solve(g,b,lam):
        return np.linalg.solve(g+lam*np.diag(g.diagonal().clip(1e-6)),b)
    prior=solve(pool[:width,:width],pool[:width,width:],.85)
    weights=[]
    for days in records.transpose(1,2,0,3,4):
        local=[]
        for g in days:
            avg=g.mean(0)
            V=np.eye(width)[:,32:]
            V[:32]=-solve(avg[:32,:32],avg[:32,32:width],.85)
            gram=V.T@(avg[:width,:width]+.60*pool[:width,:width])@V
            grad=V.T@(g[:,:width,width:]-g[:,:width,:width]@prior)
            b=grad.mean(0)
            b*=abs(b)/(abs(b)+grad.std(0)/np.sqrt(end)+1e-7)
            local.append(V@solve(gram,b,.85))
        weights.append(local)
    bases,mix=geometries(end)
    return dict(weights=np.array(weights,dtype='float32'),basis=bases,mix=mix)


def signals(raw,t,st):
    x=parent.features(raw,t)
    result=[]
    for i,(u,m,w) in enumerate(zip(st['basis'],st['mix'],st['weights'])):
        rv=parent.rotate(x,u)
        ret=np.einsum('kb,bch->kch',m,w,optimize=True)
        latent=np.einsum('tkc,kch->tkh',rv,ret,optimize=True)
        forecast=np.einsum('tkh,sk->tsh',latent,u,optimize=True)
        result.append(forecast)
        if i==2:
            local=np.einsum('tkh,sk->tsh',latent*m[:,2,None],u,optimize=True)
    return trial.clock.center(np.concatenate((*result,local,result[2]-result[1],result[5][:,:,1:]),2))


def historical_design(d):
    st=state(max(4,d-d%2))
    cache={}
    def build(raw,p,t):
        key=len(raw),t.tobytes()
        if key not in cache:
            cache[key]=signals(raw,t,st)
        return np.concatenate((trial.clock.center(p[t]),cache[key]),2)
    return build


def fit(rows):
    edges=np.cumsum([3]+[3]*9+[2])
    sets=[(np.arange(a,b),3,.42) for a,b in zip(edges,edges[1:])]
    return dict(**parent.clocks.linear.gram(rows,sets),**state(len(rows)+4))


changes=parent.changes


def correction():
    with np.load(OUT/'model.npz') as f:
        st=dict(f)
    def apply(raw,p):
        z=signals(raw,np.arange(len(raw)),st)
        return changes(np.concatenate((trial.clock.center(p),z),2),st)[int(st['pick'])]
    return apply


def layer():
    return study.layer(sys.modules[__name__],parent)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
