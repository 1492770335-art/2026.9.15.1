"""Forecast delayed returns through forward and reversed transitions between book fields."""
from functools import lru_cache
import json
import sys
import numpy as np
import torch
from scipy.linalg import solve_triangular

import y3_peer_carry_impulse_references as history
import experiment_joint_factor_order_calibration_v1 as replay
import experiment_y3_settled_factor_confidence_v1 as observed

parent,trial,stack=history.incumbent,history.trial,history.stack
OUT=trial.clock.ART/'y3_directed_book_dynamics_v1'
ARMS=('unchanged','direct far regression','forward transport','antisymmetric transport',
      'direct and forward','symmetric and antisymmetric','separate transition lengths','backward transport',
      'dynamics and payoff on separate days','two-step transition','direct and cross-fitted transport')
GROUPS=((),(0,),(1,),(3,),(0,1),(3,4),(5,6),(2,),(7,),(8,),(0,7))


def field_state(raw,t):
    cols=[]
    stop=t+1
    lo=np.where(t>=2401,2401,0)
    for i in range(0,100,20):
        v=np.tanh(.65*np.arcsinh(raw[:,:,i:i+20].astype('float32')))
        path=np.concatenate((np.zeros_like(v[:1],dtype=float),v.cumsum(0,dtype=float)))
        def average(span):
            start=np.maximum(lo,stop-span)
            return (path[stop]-path[start])/(stop-start)[:,None,None]
        quick,book,slow=[average(h) for h in (4,20,128)]
        # Use innovations so nearly constant stock identities cannot dominate the transition.
        cols.append(trial.clock.center(np.concatenate((quick-book,book-slow),axis=2)).astype('float32'))
    return np.concatenate(cols,axis=2)


@lru_cache(maxsize=16)
def record(d):
    raw=trial.clock.book.rawday(d)
    y=trial.clock.labels(d).reshape(-1,50,3)
    t=np.arange(128+d%13,len(raw)-60,13)
    t=t[(t+60<2401)|(t>=2529)]
    at=np.unique(np.concatenate((t,t+20,t+40)))
    fields=field_state(raw,at)
    x,a,b=[fields[np.searchsorted(at,t+h)].reshape(-1,fields.shape[-1]).astype(float) for h in (0,20,40)]
    pay=trial.clock.center(np.stack([y[t+h,:,1] for h in (0,20,40)],axis=2)).reshape(-1,3).clip(-4,4)
    return (x.T@x+a.T@a+b.T@b)/(3*len(x)),np.stack((x.T@a,x.T@b))/len(x),x.T@pay/len(x)


def heads(end):
    gs,cs,bs=[np.stack(a) for a in zip(*(record(d) for d in range(end)))]
    g=gs.mean(0)
    root=np.linalg.cholesky(g+.45*np.diag(g.diagonal().clip(.002))+.0005*np.eye(len(g)))
    wh=lambda a:solve_triangular(root,a,lower=True)
    wr=np.stack([wh(b) for b in bs])
    maps=np.stack([[wh(wh(c).T).T for c in day] for day in cs])
    def reward(x):
        mu=x.mean(0)
        return mu*abs(mu)/(abs(mu)+x.std(0)/np.sqrt(len(x))+1.e-6)
    ret=reward(wr)
    one,two=maps.mean(0)
    forward=(one+two)@ret[:,0]
    reverse=(one.T+two.T)@ret[:,0]
    separated=.5*sum(maps[j::2].mean(0).sum(0)@reward(wr[1-j::2])[:,0] for j in (0,1))
    cols=(ret[:,1:].sum(1),forward,reverse,.5*(forward-reverse),.5*(forward+reverse),
          one@ret[:,0],two@ret[:,0],separated,(one+one@one)@ret[:,0])
    return solve_triangular(root.T,np.stack(cols,axis=1),lower=False)


def stats(p,y,t,advice):
    ratio=observed.historical_geometry()[1][2,1]/observed.historical_geometry()[1][1,1]
    far=ratio*(y[t+20,:,1]+y[t+40,:,1])
    z=np.concatenate((np.ones_like(far[:,:,None]),p[t],advice,far[:,:,None],y[t,:,2:3]),axis=2)
    z=z.reshape(-1,z.shape[-1]).astype(float)
    return z.T@z


def weights(rows,groups=GROUPS):
    days=np.stack([np.mean(d,axis=0) for d in rows])
    c=stack.covariance(days.sum(0))
    z=np.eye(len(c)-2)[:,3:]
    z[:3]=-np.linalg.solve(c[:3,:3]+.12*np.diag(c.diagonal()[:3]),c[:3,3:-2])
    g=z.T@c[:-2,:-2]@z
    def grad(m):
        cov=stack.covariance(m)
        beta=np.clip(cov[2,-1]/cov[2,2],.35,3)
        return z.T@(cov[:-2,-1]/beta-cov[:-2,2])
    mu=grad(days.sum(0))
    day=np.stack([grad(m)/m[0,0] for m in days])
    mu*=abs(day.mean(0))/(abs(day.mean(0))+day.std(0)/np.sqrt(len(day))+1.e-8)
    out=[]
    for ids in groups:
        w=np.zeros(z.shape[1])
        if ids:
            ix=list(ids)
            gram=g[np.ix_(ix,ix)]
            w[ix]=(.65*np.linalg.solve(gram+.7*np.diag(gram.diagonal().clip(1.e-7)),mu[ix])).clip(-2.5,2.5)
        out.append(z@w)
    return np.stack(out)


def gains(m,bank):
    c=stack.covariance(m)
    v=bank.copy()
    v[:,2]+=1
    corr=(v@c[:-2,-2:])/np.sqrt(((v@c[:-2,:-2]*v).sum(1)*c[-1,-1])[:,None])
    return corr-corr[0]


def train():
    OUT.mkdir(exist_ok=True)
    rows,folds,daily=[],[],[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        refs=history.references(d,raw) if d>=8 else [history.study.screen.history.market.prior.path.early.load(d,raw)]
        if d%2==0:
            h=heads(d)
            if d>=8:
                w=weights(rows)
                group=[]
        t=np.arange(d%5,len(raw)-60,5)
        t=t[(t+60<2401)|(t>=2401)]
        a=field_state(raw,t)@h
        if d==8:
            ix=t<2477
            np.testing.assert_allclose(a[ix],field_state(raw[:2477],t[ix])@h,rtol=0,atol=1.e-6)
        grams=[stats(p,y,t,a) for p in refs]
        rows.append(grams)
        if d>=8:
            group.append(grams)
            daily.append([gains(m,w) for m in grams])
            if d%2:
                folds.append([gains(m,w) for m in np.sum(group,0)])
    f,dd=np.array(folds),np.array(daily)
    a=f.mean(0)
    good=[j for j in (2,3,4,5,6,8,9,10) if a[:,j].min()>0 and (a[:,j,1]>a[:,1,1]).all()
          and (a[:,j,1]>a[:,7,1]).all() and f[-1,:,j,1].min()>0 and (dd[:,:,j,1]>0).sum(0).min()>=5
          and (f[:,:,j,0]>0).sum(0).min()>=3]
    pick=max(good,key=lambda j:a[:,j,1].min()) if good else 0
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,arms=ARMS,metrics=['far contribution','Y3 gain'],
                                                   mean=a.tolist(),folds=f.tolist(),daily=dd.tolist())))
    print(ARMS[pick],a.round(6),flush=True)
    if pick:
        w=weights(rows)[pick]
        np.savez(OUT/'model.npz',book=heads(16)@w[3:],forecast=w[:3],pick=pick)
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        w,b=f['book'],f['forecast']
    def apply(raw,p):
        parts=[]
        for t in np.array_split(np.arange(len(raw)),max(1,(len(raw)+999)//1000)):
            parts.append(field_state(raw,t)@w+p[t]@b)
        return np.concatenate(parts)
    return apply


def layer():
    return history.study.layer(sys.modules[__name__],parent)


def replay_with():
    change=correction()
    def read(d,raw):
        p,h,n=history.read(d,raw)
        q=p.copy();part=h.copy()
        q[:,:,2]+=change(raw,p);part[:,:,2]+=change(raw[:n],h)
        return q,part,n
    return stack.verifier,read


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['replay'] or train():
        replay.replay(sys.modules[__name__])
