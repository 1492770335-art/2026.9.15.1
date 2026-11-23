"""Forecast the aftermath of the recurring price pulse once its direction is observable."""
import argparse
import json
import numpy as np
import experiment_recurring_boundary_payoff_v1 as before
import experiment_repricing_response_joint_v1 as joint
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'observed_boundary_response_v1'
LENGTH=128


def features(raw,t,b):
    r=raw[:,:,0].astype(float)-.00136
    price=np.vstack((np.zeros(50),r.cumsum(0)))
    age=t-b
    end=t+1
    cap=np.minimum(end,b+4)
    vol=np.sqrt(.15+(r[b-160:b]**2).mean(0))
    pulse=(price[cap]-price[b])/vol/2
    after=(price[end]-price[cap])/np.sqrt(np.maximum(end-cap,1))[:,None]/vol
    before_move=np.broadcast_to((price[b]-price[b-20])/np.sqrt(20)/vol,pulse.shape)
    def pressure(ids):
        return (np.arcsinh(raw[ids,:,41].astype(float))-np.arcsinh(raw[ids,:,42].astype(float))+
                np.arcsinh(raw[ids,:,54].astype(float))-np.arcsinh(raw[ids,:,55].astype(float)))*.5
    flow=pressure(t)
    jump_flow=pressure(cap-1)-pressure(b-1)
    fast=np.exp(-age/8.)[:,None]
    slow=np.exp(-age/45.)[:,None]
    components=np.stack((np.tanh(pulse),np.arcsinh(pulse)*.6,np.tanh(after),
                         np.tanh(before_move),np.tanh(flow),np.tanh(jump_flow),
                         np.tanh(pulse)*np.tanh(flow*pulse),
                         np.tanh(pulse)*np.tanh(after*pulse)),axis=2)
    m=components.mean(1,keepdims=True)
    return [np.concatenate((s,s*fast[:,:,None],s*slow[:,:,None]),axis=2)
            for s in (components-m,m)]


def stats(raw,y,b):
    t=np.arange(b,b+LENGTH)
    m=y[t].mean(1,keepdims=True)
    result=[]
    for x,z in zip(features(raw,t,b),(y[t]-m,m)):
        a=x.reshape(-1,x.shape[-1]);v=z.reshape(-1,3)
        result.append((a.mean(0),v.mean(0),a.T@a/len(a),a.T@v/len(a)))
    return result


def record(day):
    raw=before.book.rawday(day)
    y=labels(day).reshape(-1,50,3)
    placebo=[stats(raw,y,b) for b in before.CONTROLS]
    avg=[[np.mean(v,axis=0) for v in zip(*part)] for part in zip(*placebo)]
    return avg,stats(raw,y,before.BOUNDARY)


def changes(raw,t,model):
    use=(t>=before.BOUNDARY)&(t<before.BOUNDARY+LENGTH)
    out=[np.zeros((len(t),50,3),np.float32),np.zeros((len(t),1,3),np.float32)]
    if not use.any():return out
    for i,(x,z) in enumerate(zip(features(raw,t[use],before.BOUNDARY),out)):
        prediction=[(x-a[i][0])@a[i][1] for a in model]
        delta=.25*(prediction[1]-prediction[0]).clip(-2,2)
        if i==0:delta-=delta.mean(1,keepdims=True)
        z[use]=delta
    return out


def load():
    parent=joint.load()
    with np.load(OUT/'model.npz') as st:
        model=[[(st[f'c{i}{j}'],st[f'w{i}{j}']) for j in range(2)] for i in range(2)]
        keep=st['mask']
    def predict(raw,p):
        q=parent(raw,p).copy()
        for z,k in zip(changes(raw,np.arange(len(raw)),model),keep):q+=z*k
        return q
    return predict


def screen():
    OUT.mkdir(exist_ok=True)
    history,folds,daily=[],[],[]
    for end in (8,10,12,14):
        history.extend(record(d) for d in range(len(history),end))
        model=before.fit(history)
        rows=[[],[]]
        for d in (end,end+1):
            raw=before.book.rawday(d)
            t=np.arange(len(raw))
            y=labels(d).reshape(-1,50,3)
            p=np.load(before.parent_model.block.segment.best.refit.OUT/f'reference_{d}.npy',mmap_mode='r').copy()
            p[:,:,1]=np.load(before.parent_model.linear.earlier.OUT/f'y2_day_{d}.npy',mmap_mode='r')
            deltas=changes(raw,t,model)
            for n in (before.BOUNDARY,before.BOUNDARY+1,before.BOUNDARY+22):
                for a,b in zip(changes(raw[:n],t[:n],model),deltas):
                    np.testing.assert_allclose(a,b[:n],rtol=0,atol=2e-6)
            for g,z in zip(rows,deltas):g.append(tuple(a.reshape(-1,3) for a in (p,p+z,y)))
            daily.append([before.parent_model.linear.gains(g[-1:]) for g in rows])
        folds.append([before.parent_model.linear.gains(g) for g in rows])
        print(end+1,'IC gains local/market',folds[-1],flush=True)
    keep=(np.min(folds,axis=0)>0)&((np.array(daily)>0).sum(0)>=5)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,retained=keep.tolist()),indent=2))
    if keep.any():
        history.extend(record(d) for d in range(len(history),16))
        data=dict(mask=keep)
        for i,part in enumerate(before.fit(history)):
            for j,(c,w) in enumerate(part):data.update({f'c{i}{j}':c,f'w{i}{j}':w})
        np.savez(OUT/'model.npz',**data)


def replay():
    if not np.any(json.loads((OUT/'results.json').read_text())['retained']):return
    baseline=json.loads((joint.OUT/'reload_verification.json').read_text())['pooled']['score']
    for name in ('raw_replay','reload_verification'):
        res=before.parent_model.linear.best.online.best.roll.gls.best.walk.replay(load())
        if name=='reload_verification':
            old=json.loads((OUT/'raw_replay.json').read_text())
            np.testing.assert_allclose(res['pooled']['candidate'],old['pooled']['candidate'],rtol=0,atol=1e-10)
        (OUT/f'{name}.json').write_text(json.dumps(res,indent=2))
        print(name,res['pooled']['score'],flush=True)
        if res['pooled']['score']<=baseline:break


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--replay',action='store_true')
    if parser.parse_args().replay:replay()
    else:screen()
