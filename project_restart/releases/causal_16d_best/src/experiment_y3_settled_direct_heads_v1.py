"""Update interval-head prices intraday from already settled outcomes and the live reference forecasts."""
import json
import sys

import numpy as np
import torch

import experiment_compounded_return_audit_v1 as compound
import experiment_y3_disjoint_forecast_stack_v1 as historical
import experiment_y3_disjoint_peer_gru_v1 as net
import y3_disjoint_validation as trial

OUT=trial.clock.ART/'y3_settled_direct_heads_v1'
ARMS=['fixed both','settled opposite stocks','settled own stocks','matched half agreement',
      'reassigned returns','delayed calibration','joint Y3 and late alignment','half opposite stocks']
SECONDARY='last forty and disjoint twenty tick stock-relative correlation'


def settlement(raw):
    k,coeff,row=trial.curve.settled.historical_geometry()
    n=len(raw);ix=np.arange(n)+1
    log=compound.compound(raw,k)
    missing=np.vstack((np.zeros((1,50),int),(~compound.present(raw,row)).cumsum(0)))
    returns=[]
    for width in (60,40):
        start=(ix-width).clip(0)
        val=coeff[2,1]*np.expm1(log[ix]-log[start])/k
        returns.append(val.astype('float32'))
    valid=(missing[ix]==missing[(ix-60).clip(0)])&(ix>=61)[:,None]
    return np.stack(returns,2),valid


def state(raw):
    return net.state(raw),settlement(raw)


def calibration(h,p,settled,shift,delay):
    y,valid=settled
    block=80;length=len(h)
    g=np.zeros((2,6,6))
    q=np.zeros((length,50,4),dtype='float32')
    ratio=trial.curve.settled.historical_geometry()[1]
    base=np.stack([p[:,:,2],p[:,:,2]-p[:,:,1]*(ratio[2,1]/ratio[1,1])],2)
    for t in range(0,length,block):
        last=t-60-delay
        if last<0:
            continue
        origins=np.arange(max(0,last-block+1),last+1,7)
        end=origins+60+delay
        # The delayed control pairs each origin with a later realized outcome that has already matured.
        truth=np.roll(y[end],shift,axis=1)
        mask=np.roll(valid[end],shift,axis=1)
        slopes=[]
        for side in (0,1):
            xx=np.concatenate([h[origins,side::2],base[origins,side::2],truth[:,side::2]],axis=2).astype(float)
            keep=mask[:,side::2]
            # Remove contemporaneous common moves within the training panel.
            xx=(xx-(xx*keep[:,:,None]).sum(1,keepdims=True)/keep.sum(1).clip(1)[:,None,None])*keep[:,:,None]
            v=xx.reshape(-1,6)
            g[side]=.88*g[side]+(v.T@v)/len(v)
            s=g[side]
            k=(np.array([s[2,4],s[3,5]])+np.array([.15,.08]))/(s.diagonal()[2:4]+np.array([.15,.08]))
            b=s[:2,4:6]/k.clip(.35,2.5)-s[:2,2:4]
            prior=12*np.diag(s.diagonal()[:2].clip(.0005))+.012*np.eye(2)
            w=np.linalg.solve(s[:2,:2]+prior,b).clip(-.55,.55)
            slopes.append(w)
        for side in (0,1):
            v=h[t:t+block,side::2]
            own,other=slopes[side],slopes[1-side]
            agree=(own[:,0]*other[:,0]>0)
            horizons=(other[:,0]*other[:,1]>0)
            w=np.stack([other[:,0],own[:,0],.5*(other[:,0]+own[:,0])*agree,other[:,0]*horizons],axis=1)
            q[t:t+block,side::2]=v@w
    return trial.clock.center(q)


def predictions(h,p,settled):
    correct=calibration(h,p,settled,0,0)
    wrong=calibration(h,p,settled,16,0)[:,:,0]
    delay=calibration(h,p,settled,0,80)[:,:,0]
    return [.15*h.sum(2),correct[:,:,0],correct[:,:,1],correct[:,:,2],wrong,delay,correct[:,:,3],.5*correct[:,:,0]]


def changes_with_reference(s,m,p):
    z,settled=s
    return predictions(historical.heads(z,m),p,settled)


def select(f,d):
    f,d=np.asarray(f),np.asarray(d)
    def passes(a,b=None,idx=(0,1,2,3)):
        v,h=f[...,a],d[...,a]
        if b is not None:
            v,h=v-f[...,b],h-d[...,b]
        return trial.scoring.select(v[:,:,list(idx),None],h[:,list(idx),None]) is not None
    ids=[i for i in (1,2,3,6,7) if passes(i) and all(passes(i,j,(0,)) for j in (0,4,5))]
    return int(max(ids,key=lambda i:f[:,0,0,i].mean())) if ids else None


def run():
    OUT.mkdir(exist_ok=True)
    folds=[];daily=[]
    for d in range(8,16):
        raw=trial.clock.book.rawday(d);y=trial.clock.labels(d).reshape(-1,50,3)
        settled=settlement(raw)
        with np.load(historical.CACHE/f'day_{d:02d}.npz') as cache:
            h=cache['h']
            assert int(cache['train_end'])==d-d%2 and cache['raw_hash'].item()==trial.previous.prior.digest(raw)
        if d%2==0:
            stats=[[],[]]
        for j,p in enumerate(trial.refs(d,raw)):
            delta=predictions(h,p,settled)
            if d%2==0 and j==0:
                for a,b in zip(delta,predictions(h[:367],p[:367],settlement(raw[:367]))):
                    np.testing.assert_allclose(a[:367],b,atol=2e-6,rtol=0)
            m=[q.sum(0) for q in trial.statistics(p,y,delta)]
            stats[j].append(m)
            if j==0:
                daily.append([trial.scoring.audit.pooled(q).tolist() for q in m])
        if d%2:
            folds.append([[trial.scoring.audit.pooled(q).tolist() for q in np.sum(v,axis=0)] for v in stats])
    pick=select(folds,daily)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=pick,folds=folds,daily=daily),indent=2))
    print('selected',pick,flush=True)
    if pick is not None:
        history=[]
        for d in range(16):
            raw=trial.clock.book.rawday(d)
            history.append(net.observe(net.state(raw),trial.clock.labels(d).reshape(-1,50,3),d))
        model,_=net.fit(history)
        np.savez(OUT/'model.npz',pick=pick,**model)
        p=trial.previous
        trial.scoring.outer(sys.modules[__name__],reference=trial.parent,preceding=(p.prior.prior.incumbent,p.prior.incumbent,p.incumbent))


def correction():
    return trial.previous.prior.correction(sys.modules[__name__])


def layer():
    return trial.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
