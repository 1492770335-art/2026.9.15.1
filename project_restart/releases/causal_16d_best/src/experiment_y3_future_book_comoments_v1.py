"""Forecast joint changes in signed fields and activity before pricing their later payoff."""
from functools import lru_cache
import json
import sys
import numpy as np
import torch
from scipy.linalg import solve

import experiment_y3_directed_book_dynamics_v1 as model
import experiment_y3_peer_arrival_alignment_v1 as arrival

history,parent,stack,trial=model.history,model.parent,model.stack,model.trial
OUT=trial.clock.ART/'y3_future_book_comoments_v1'
ARMS=('unchanged','direct late forecast','first moments','conditional independence','joint moment forecast',
      'conditional covariance alone','mean and covariance','shuffled future partners','cross-date joint forecast',
      'direct forecast plus conditional covariance','cross-date covariance alone')
GROUPS=((),(0,),(1,),(2,),(3,),(4,),(1,4),(5,),(6,),(0,4),(7,))


def interactions(a,b):
    return (a[:,:,:,None]*b[:,:,None,:]).reshape(*a.shape[:2],64)


def observations(raw,t,future=False):
    signed=arrival.signals(raw)
    amount=np.tanh(.65*abs(np.arcsinh(raw[:,:,[0,7,11,14,21,38,54,77]].astype('float32'))))
    fields=np.concatenate((signed,amount),axis=2)
    path=np.concatenate((np.zeros_like(fields[:1],dtype=float),fields.cumsum(0,dtype=float)))
    stop=t+1
    def avg(lo,hi):
        return ((path[hi]-path[lo])/(hi-lo)[:,None,None]).astype('float32')
    recent,medium,slow=[avg(np.maximum(stop-h,np.where(t>=2401,2401,0)),stop) for h in (4,20,96)]
    now=fields[t]
    delta=medium-slow
    x=np.concatenate((np.ones_like(now[:,:,:1]),now-recent,recent-medium,delta,slow,
                      interactions(recent[:,:,:8],recent[:,:,8:]),
                      interactions(delta[:,:,:8],slow[:,:,8:])),axis=2)
    if not future:
        return x,None,None
    later=avg(stop,stop+20)-medium
    a,b=later[:,:,:8],later[:,:,8:]
    ab=interactions(a,b)
    wrong=interactions(a,np.roll(b,17,axis=1))
    return x,np.concatenate((a,b,ab),axis=2),np.concatenate((a,np.roll(b,17,axis=1),wrong),axis=2)


@lru_cache(maxsize=16)
def moments(d):
    raw=trial.clock.book.rawday(d)
    y=trial.clock.labels(d).reshape(-1,50,3)
    t=np.arange(90+d%13,len(y)-60,13)
    t=t[(t+60<2401)|(t>=2491)]
    x,f,wrong=observations(raw,t,True)
    x,f,wrong=[v.reshape(-1,v.shape[-1]).astype(float) for v in (x,f,wrong)]
    pay=trial.clock.center(y[t+20,:,1]+y[t+40,:,1]).reshape(-1).clip(-5,5)
    n=len(x)
    return (x.T@x/n,x.T@pay/n,
            np.stack((x.T@f,x.T@wrong))/n,np.stack((f.T@f,wrong.T@wrong))/n,
            np.stack((f.T@pay,wrong.T@pay))/n)


def ridge(g,b):
    return solve(g+.55*np.diag(g.diagonal().clip(.0001))+.00005*np.eye(len(g)),b,assume_a='pos')


def params(end):
    days=[moments(d) for d in range(end)]
    out={}
    for name,subset in (('full',days),('even',days[::2]),('odd',days[1::2])):
        gram,ret,target,field,cov=[np.mean(v,axis=0) for v in zip(*subset)]
        for j,suffix in enumerate(('', '_wrong') if name=='full' else ('',)):
            out[name+suffix+'_state']=ridge(gram,target[j]).astype('float32')
            out[name+suffix+'_pay']=ridge(field[j],cov[j]).astype('float32')
        if name=='full':
            out['direct']=ridge(gram,ret).astype('float32')
    return out


def advice(raw,t,st):
    x=observations(raw,t)[0]
    def predict(source,price):
        f=x@st[source+'_state']
        w=st[price+'_pay']
        marginal=f[:,:,:16]@w[:16]
        independent=interactions(f[:,:,:8],f[:,:,8:16])@w[16:]
        joint=f[:,:,16:]@w[16:]
        return marginal,marginal+independent,marginal+joint,joint-independent
    a,b,c,k=predict('full','full')
    wrong=predict('full_wrong','full_wrong')[2]
    l,r=predict('even','odd'),predict('odd','even')
    return trial.clock.center(np.stack((x@st['direct'],a,b,c,k,wrong,.5*(l[2]+r[2]),.5*(l[3]+r[3])),axis=2))


CANDIDATES=(4,5,6,8,9,10)
CONTROLS=(1,3,7)


def train(mod=None):
    mod=mod or sys.modules[__name__]
    history=mod.history
    mod.OUT.mkdir(exist_ok=True)
    rows,folds,daily=[],[],[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        refs=history.references(d,raw) if d>=8 else [history.study.screen.history.market.prior.path.early.load(d,raw)]
        if d%2==0:
            st=mod.params(d)
            if d>=8:
                w=model.weights(rows,mod.GROUPS);group=[]
        t=np.arange(d%5,len(raw)-60,5)
        t=t[(t+60<2401)|(t>=2401)]
        z=mod.advice(raw,t,st)
        if d==8:
            ix=t<2477
            np.testing.assert_allclose(z[ix],mod.advice(raw[:2477],t[ix],st),rtol=0,atol=1.e-6)
        grams=[model.stats(p,y,t,z) for p in refs]
        rows.append(grams)
        if d>=8:
            group.append(grams);daily.append([model.gains(m,w) for m in grams])
            if d%2: folds.append([model.gains(m,w) for m in np.sum(group,0)])
    f,dd=np.array(folds),np.array(daily)
    a=f.mean(0)
    valid=[j for j in mod.CANDIDATES if a[:,j].min()>0 and all((a[:,j,1]>a[:,k,1]).all() for k in mod.CONTROLS)
           and (f[-1,:,j,1]>0).all() and (dd[:,:,j,1]>0).sum(0).min()>=5 and (f[:,:,j,0]>0).sum(0).min()>=3]
    pick=max(valid,key=lambda j:a[:,j,1].min()) if valid else 0
    (mod.OUT/'results.json').write_text(json.dumps(dict(selected=pick,arms=mod.ARMS,metrics=['far contribution','Y3 gain'],folds=f.tolist(),daily=dd.tolist(),mean=a.tolist())))
    print(mod.ARMS[pick],a.round(6),flush=True)
    if pick:
        np.savez(mod.OUT/'model.npz',**mod.params(16),w=model.weights(rows,mod.GROUPS)[pick],pick=pick)
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f: st=dict(f)
    def apply(raw,p):
        parts=[]
        for t in np.array_split(np.arange(len(raw)),max(1,(len(p)+799)//800)):
            z=advice(raw,t,st)
            parts.append(p[t]@st['w'][:3]+z@st['w'][3:])
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
        model.replay.replay(sys.modules[__name__])
