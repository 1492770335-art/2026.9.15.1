"""Calibrate the size of predicted return bands from the energy of predictable, unsigned book states."""
import json
import sys

import numpy as np

import experiment_y3_omitted_field_persistence_v1 as source

parent,history,base=source.parent,source.history,source.base
OUT=parent.OUT.parent/'joint_field_state_amplitudes_v1'
ARMS=('unchanged','core magnitude','instantaneous omitted magnitude','reassigned remote magnitudes','persistent book magnitude',
      'remote excess above market','market-wide state energy','late-band adjustment only','near-band adjustment only')
SCALE=np.array([.160943/.558942,base.RATIO,1.])


def band(p):
    return np.diff(p*SCALE,axis=2,prepend=np.zeros_like(p[:,:,:1]))


def snapshot(raw,t,st):
    w,bank,cw=st
    c,x,_=source.views(raw,t)
    r=x-c@w
    core,instant,far=[np.abs(v).clip(0,4) for v in (c@cw,r@bank[:,:,-1],r@bank[:,:,2])]
    m=far.mean(1,keepdims=True)
    return np.stack((core,instant,np.roll(far,13,1),far,far-m,np.broadcast_to(m,far.shape)),2).astype(np.float32)


def representation(raw,t,st):
    out=[]
    for lo,hi in base.segments(len(raw)):
        use=t[(t>=lo)&(t<hi)]
        for start in range(0,len(use),180):
            ticks=use[start:start+180]
            a=max(lo,int(ticks[0])-79)
            b=int(ticks[-1])+1
            out.append(snapshot(raw[a:b],ticks-a,st))
    return np.concatenate(out)


def moments(x,p,y):
    # One forecast-band slope multiplies one centered return forecast. A common scalar adjustment cancels in Pearson.
    a,b=band(p),band(y)
    a-=a.mean(1,keepdims=True)
    b-=b.mean(1,keepdims=True)
    gram=[];cross=[]
    for j in range(3):
        q,truth=a[:,:,j],b[:,:,j]
        slope=np.sum(q*truth)/np.sum(q*q).clip(1.e-7)
        err=truth/slope.clip(.35,2.8)-q
        xx=(x*q[:,:,None,None]).transpose(2,0,1,3).reshape(x.shape[2],-1,x.shape[3]).astype(float)
        gram.append(xx.transpose(0,2,1)@xx/xx.shape[1])
        cross.append((xx.transpose(0,2,1)@err.ravel())/xx.shape[1])
    return np.stack(gram),np.stack(cross)


def fit(end,st):
    rows=[];mean=[]
    for d in range(4,end):
        raw,ps,y=source.read_day(d)
        t=np.arange(d%9,len(y),9)
        x=representation(raw,t,st)
        center=x.mean((0,1),dtype=float).astype(np.float32)
        mean.append(center)
        x-=center
        obs=[moments(x,p[t],y[t]) for p in ps]
        rows.append(tuple(np.mean(v,0) for v in zip(*obs)))
    gs,bs=[np.array(v) for v in zip(*rows)]
    g=gs.mean(0);b=bs.mean(0)
    b*=abs(b)/(abs(b)+bs.std(0)/np.sqrt(len(rows))+.000002)
    w=.32*np.linalg.solve(g+1.3*np.eye(g.shape[-1])*g.diagonal(axis1=-2,axis2=-1).clip(.00002)[:,:,:,None],b[...,None])[...,0]
    return np.mean(mean,0),w


def changes(x,p,mu,w):
    a=band(p)
    a-=a.mean(1,keepdims=True)
    fields=x-mu
    ratios=np.einsum('tsaf,haf->tsah',fields,w).clip(-.35,.35)
    bands=ratios*a[:,:,None,:]
    bands-=bands.mean(1,keepdims=True)
    late=bands[:,:,3:4].copy()
    late[:,:,:,:2]=0
    early=bands[:,:,3:4].copy()
    early[:,:,:,2]=0
    return (np.concatenate((bands,late,early),2).cumsum(3)/SCALE).astype(np.float32)


def score(p,y,z):
    grams=[]
    for j in range(3):
        v=np.concatenate((np.ones_like(p[:,:,:1]),p[:,:,j:j+1],z[:,:,:,j],y[:,:,j:j+1]),2)
        v=v.reshape(-1,v.shape[2]).astype(float)
        grams.append(v.T@v)
    return np.stack(grams)


def gain(x):
    c=base.covariance(x)
    baseline=c[0,-1]/np.sqrt(c[0,0]*c[-1,-1])
    corr=(c[0,-1]+c[1:-1,-1])/np.sqrt((c[0,0]+2*c[0,1:-1]+c.diagonal()[1:-1])*c[-1,-1])
    return np.r_[0.,corr-baseline]


def train():
    OUT.mkdir(exist_ok=True)
    rows=[source.stats(d) for d in range(16)]
    folds,daily=[],[]
    for end in (8,12):
        state=source.axes(rows[:end])
        mu,w=fit(end,state)
        group=[]
        for d in range(end,end+4):
            raw,ps,y=source.read_day(d)
            x=representation(raw,np.arange(len(raw)),state)
            if d==8:
                n=409
                np.testing.assert_allclose(x[:n],representation(raw[:n],np.arange(n),state),atol=1e-6,rtol=0)
            pair=[score(p,y,changes(x,p,mu,w)) for p in ps]
            group.append(pair)
            daily.append([[gain(m) for m in v] for v in pair])
        folds.append([[gain(m) for m in v] for v in np.sum(group,0)])
    f,ds=np.array(folds),np.array(daily)
    picks=[]
    for j in range(3):
        ok=[i for i in range(4,len(ARMS)) if f[:,:,j,i].min()>0 and (ds[:,:,j,i]>0).sum(0).min()>=5 and
            (f[:,:,j,i].mean(0)>f[:,:,j,1:4].mean(0).max(1)).all()]
        picks.append(max(ok,key=lambda i:f[:,:,j,i].min()) if ok else 0)
    (OUT/'results.json').write_text(json.dumps(dict(selected=picks,arms=ARMS,folds=f.tolist(),daily=ds.tolist(),windows=[[9,12],[13,16]]),indent=2))
    print('selected',[ARMS[i] for i in picks],f[:,0,2].round(6).tolist(),flush=True)
    if any(picks):
        state=source.axes(rows)
        mu,w=fit(16,state)
        np.savez(OUT/'model.npz',s0=state[0],s1=state[1],s2=state[2],mu=mu,w=w,pick=picks)
    return any(picks)


def correction():
    with np.load(OUT/'model.npz') as f:
        st=[f[f's{i}'] for i in range(3)]
        w,mu,picks=f['w'],f['mu'],f['pick']
    def update(raw,p):
        x=representation(raw,np.arange(len(p)),st)
        z=changes(x,p,mu,w)
        return z[:,:,np.maximum(picks-1,0),np.arange(3)]*(picks>0)
    return update


def layer():
    before,update=parent.layer(),correction()
    def apply(raw,initial):
        p=before(raw,initial)
        return p+update(raw,p)
    return apply


def replay_with():
    update=correction()
    def read(d,raw):
        p,h,n=history.read(d,raw)
        return p+update(raw,p),h+update(raw[:n],h),n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
