"""Share a lagged price-impact curve across twelve five-tick forward returns."""
import json
import numpy as np
import torch

import experiment_y3_disjoint_book_curve_v1 as source
import experiment_y3_peer_arrival_alignment_v1 as arrival

curve,clock,incumbent=source.curve,source.clock,source.incumbent
OUT=clock.ART/'y3_finite_impact_curve_v1'
MEMORY=12
HORIZONS=12
STEP=5


def averages(raw):
    x=arrival.signals(raw)
    right=np.arange(len(x))+1
    sums=np.concatenate((np.zeros_like(x[:1],dtype=float),x.cumsum(0,dtype=float)))
    left=(right-STEP).clip(0)
    return ((sums[right]-sums[left])/(right-left)[:,None,None]).astype('float32')


def autoregression():
    stats=[]
    for d in range(8):
        x=averages(clock.book.rawday(d))
        t=np.arange(20+d%5,len(x),5)
        stats.append(curve.gram(np.concatenate((x[t-5],x[t-10],x[t-20]),2),x[t]))
    return curve.solve(*[np.mean(c,0) for c in zip(*stats)],.10).astype('float32')


def history(raw,t,ar):
    x=averages(raw)
    idx=np.arange(len(x))
    previous=np.concatenate([x[(idx-h).clip(0)] for h in (5,10,20)],2)
    e=x-previous@ar
    blocks=[e[(t-STEP*lag).clip(0)]*(t>=STEP*lag)[:,None,None] for lag in range(MEMORY)]
    return np.concatenate(blocks,2)


def blocks(y,t):
    k,coef,_=curve.settled.historical_geometry()
    growth=np.ones_like(y[t,:,0])
    out=[]
    for j in range(HORIZONS):
        r=(y[t+STEP*j,:,0]-coef[0,0])/coef[0,1]
        out.append(coef[2,1]*growth*r)
        growth=growth*(1+k*r)
    return np.stack(out,2)


def fitted(stats,matched):
    g,r=[np.mean(c,0) for c in zip(*stats)]
    size=g.shape[0]
    width=size//MEMORY
    full=size+(HORIZONS-1)*width
    gram=np.zeros((full,full));rhs=np.zeros(full)
    # Both mappings consume exactly the same observations. The control exchanges
    # the two five-tick payouts in each twenty-tick sub-band while preserving their sum.
    order=np.arange(HORIZONS) if matched else np.arange(HORIZONS).reshape(-1,4)[:,[2,3,0,1]].ravel()
    for j,k in enumerate(order):
        span=slice(j*width,j*width+size)
        gram[span,span]+=g
        rhs[span]+=r[:,k]
    penalty=np.diag(.85*gram.diagonal().clip(.0001)+.0001)
    differences=np.diff(np.eye(full//width),axis=0)
    penalty+=np.kron(differences.T@differences,np.diag(g.diagonal().reshape(MEMORY,width).mean(0)*2.5))
    kernel=np.linalg.solve(gram+penalty,rhs)
    coefficients=np.stack([kernel[j*width:j*width+size] for j in range(HORIZONS)],axis=1)
    # Undo the control permutation so both forecasts refer to the same physical periods.
    return coefficients[:,np.argsort(order)].astype('float32')


def forecasts(raw,ar,w):
    t=np.arange(len(raw))
    local=history(raw,t,ar)
    estimates=[]
    for weights in w:
        compressed=np.column_stack((weights[:,6:10].sum(1),weights[:,4:].sum(1)))
        estimates.append(local@compressed)
    return estimates


def moves(fs):
    diff=fs[1]-fs[0]
    return [.20*clock.center(diff[:,:,0]),.20*clock.center(diff[:,:,1]),.20*diff[:,:,0]]


def run():
    ar=autoregression()
    stats,folds,daily,diagnostics=[],[],[],{}
    for d in range(16):
        raw=clock.book.rawday(d)
        y=clock.labels(d).reshape(-1,50,3)
        if d>=8:
            if d%2==0:
                w=[fitted(stats,matched) for matched in (False,True)]
                groups=[[[],[],[]],[[],[],[]]]
            fs=forecasts(raw,ar,w)
            if d%2==0:
                for full,sub in zip(fs,forecasts(raw[:411],ar,w)):
                    np.testing.assert_allclose(full[:411],sub,rtol=0,atol=3e-6)
                altered=raw[:411].copy()
                altered[:,1:]=0
                for full,sub in zip(fs,forecasts(altered,ar,w)):
                    np.testing.assert_allclose(full[:411,0],sub[:,0],rtol=0,atol=3e-6)
            z=moves(fs)
            valid=np.arange(len(y)-60)
            obs=clock.center(blocks(y,valid)[:,:,6:10].sum(2))
            for j,ref in enumerate(curve.REFS):
                p=ref(d).reshape(y.shape)
                if not j:
                    a=np.column_stack([obs.ravel(),*[clock.center(v[valid,:,0]).ravel() for v in fs]])
                    c=clock.center(p[valid]).reshape(-1,3).astype(float)
                    residual=a-c@np.linalg.lstsq(c,a,rcond=None)[0]
                    diagnostics[d+1]=dict(middle=np.corrcoef(a.T)[0,1:].tolist(),
                              partial=np.corrcoef(residual.T)[0,1:].tolist())
                for g,delta in zip(groups[j],z):
                    g.append((p,y,delta))
            daily.append([curve.metric.gain(g[-1:])['y3'] for g in groups[0]])
            if d%2:
                folds.append([[curve.metric.gain(g)['y3'] for g in side] for side in groups])
                print('ending',d+1,folds[-1],flush=True)
        t=np.arange(d%6,len(y)-60,6)
        x=history(raw,t,ar)
        obs=blocks(y,t).clip(-1.8,1.8).astype('float32')
        stats.append(curve.gram(x,obs))
    a=np.array(folds);midd=np.array([s['partial'] for s in diagnostics.values()])
    keep=((a>0).sum(0).min(0)>=3)&(a.mean(0).min(0)>0)&(a[-1].min(0)>0)&((np.array(daily)>0).sum(0)>=5)
    choices=np.flatnonzero(keep) if (midd[:,1]-midd[:,0]).mean()>0 else []
    pick=int(max(choices,key=lambda k:a[:,0,k].mean())) if len(choices) else None
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,folds=folds,daily=daily,diagnostics=diagnostics,
        arms=['middle relative','far relative','whole middle'],timing=['shuffled','aligned']),indent=2))
    if pick is not None:
        np.savez(OUT/'model.npz',ar=ar,w=[fitted(stats,m) for m in (False,True)],pick=pick)
    return pick is not None


def layer():
    parent=incumbent.layer()
    with np.load(OUT/'model.npz') as s:
        ar,w,pick=s['ar'],s['w'],int(s['pick'])
    def apply(raw,p):
        q=parent(raw,p)
        q[:,:,2]+=moves(forecasts(raw,ar,w))[pick]
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    parent,update=ancestor.load(),layer()
    return lambda raw,p:update(raw,parent(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    if run():
        source.replay.verify(OUT,layer,incumbent.OUT)
