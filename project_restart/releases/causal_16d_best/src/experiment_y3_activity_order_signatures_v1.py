"""Forecast late relative returns from the order of signed book pressure and changes in activity."""
import json
import sys

import numpy as np
import torch

import y3_tail_specialist_eval as evaluator

parent,clock,curve=evaluator.parent,evaluator.clock,evaluator.curve
impact=parent.incumbent.impact
OUT=clock.ART/'y3_activity_order_signatures_v1'


def observations(raw):
    s=impact.arrival.signals(raw)
    activity=np.stack((abs(s[:,:,0]),abs(s[:,:,1]),abs(s[:,:,3:5]).mean(2),
                       abs(s[:,:,5:]).mean(2)),axis=2)
    z=np.concatenate((s,activity),axis=2)
    return np.concatenate((np.zeros_like(z[:1],dtype=float),z.cumsum(0,dtype=float)))


def signature(prefix,t):
    sums=[];ordered=[]
    for step in (4,12):
        right=t+1
        start=(right-6*step).clip(0)
        left=(start-120).clip(0)
        origin=(prefix[start]-prefix[left])/(start-left).clip(1)[:,None,None]
        seq=[]
        for k in range(6):
            b=(right-(5-k)*step).clip(0)
            a=(b-step).clip(0)
            seq.append(((prefix[b]-prefix[a])/(b-a).clip(1)[:,None,None]-origin).astype('float32'))
        seq=np.stack(seq)
        pressure,activity=seq[:,:,:,:8],seq[:,:,:,8:]
        before=seq.cumsum(0)-seq
        def pairs(a,b):
            return np.einsum('ktsc,ktsd->tscd',a,b,optimize=False).reshape(len(t),50,-1)/15
        first=pairs(before[:,:,:,:8],activity)
        second=pairs(pressure,before[:,:,:,8:])
        sums.extend((pressure.mean(0),pressure[-1]-pressure[0],activity.mean(0),first+second))
        ordered.append(first-second)
    return clock.center(np.concatenate(sums+ordered,axis=2)),sum(x.shape[2] for x in sums)


def record(prefix,t,y):
    sums=None
    for ids in np.array_split(t,max(1,(len(t)+127)//128)):
        z,n=signature(prefix,ids)
        a=z.reshape(-1,z.shape[2])
        target=evaluator.late(y[ids]).clip(-3.,3.).ravel()
        stats=(a.T@a,a.T@target)
        if sums is None:sums=[v.astype(float) for v in stats]
        else:
            for total,part in zip(sums,stats):total+=part
    return sums[0]/(len(t)*50),sums[1]/(len(t)*50),n


def fit(records):
    g,b=[np.mean([r[k] for r in records],axis=0) for k in (0,1)]
    n=records[0][2]
    symmetric=curve.solve(g[:n,:n],b[:n],.55)
    nuisance=curve.solve(g[:n,:n],g[:n,n:],.15)
    sub=np.vstack((-nuisance,np.eye(len(g)-n)))
    moment=np.array([sub.T@r[1] for r in records])
    mean=moment.mean(0);error=moment.std(0)/np.sqrt(len(moment))
    w=curve.solve(sub.T@g@sub,mean,.75)
    safe=curve.solve(sub.T@g@sub,mean*abs(mean)/(abs(mean)+error+.00001),.75)
    return np.column_stack((np.r_[symmetric,np.zeros(len(g)-n)],sub@w,sub@safe)).astype('float32')


def predict(prefix,weights):
    rows=[]
    for start in range(0,len(prefix)-1,128):
        t=np.arange(start,min(len(prefix)-1,start+128))
        rows.append(signature(prefix,t)[0]@weights)
    return np.concatenate(rows)


def candidates(z,p):
    unordered,order,stable=np.moveaxis(z,2,0)
    return [.10*(unordered+stable-evaluator.late(p)),.22*order,.22*stable]


def run():
    hist=[];records=[];folds=[];daily=[]
    for d in range(16):
        raw=clock.book.rawday(d);prefix=observations(raw);y=clock.labels(d).reshape(-1,50,3)
        if d>=8:
            if d%2==0:
                w=fit(records);mu,bank=parent.geometry(hist);groups=[[],[]]
            z=predict(prefix,w)
            if d%2==0:
                np.testing.assert_allclose(z[:353],predict(observations(raw[:353]),w),rtol=0,atol=4e-6)
            terms=parent.terms(parent.incumbent.state(raw)[1],mu,bank)
            for j,ref in enumerate(curve.REFS):
                p=ref(d).reshape(y.shape).copy();p[:,:,2]+=terms[0]+parent.candidates(terms)[2]
                moments=[s.sum(0) for s in evaluator.statistics(p,y,candidates(z,p))]
                groups[j].append(moments)
                if not j:daily.append([evaluator.audit.pooled(a).tolist() for a in moments])
            if d%2:
                folds.append([[evaluator.audit.pooled(a).tolist() for a in np.sum(g,axis=0)] for g in groups])
                print('ending',d+1,folds[-1][0],flush=True)
        records.append(record(prefix,np.arange(d%5,len(raw)-60,5),y))
        hist.append(parent.incumbent.stats(raw,y,d))
    pick=evaluator.select(folds,daily)
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,folds=folds,daily=daily,arms=[
        'partial replacement','chronological interaction','chronological interaction with date shrinkage']),indent=2))
    if pick is not None:np.savez(OUT/'model.npz',w=fit(records),pick=pick)
    return pick is not None


def correction():
    with np.load(OUT/'model.npz') as s:w,pick=s['w'],int(s['pick'])
    return lambda raw,p:candidates(predict(observations(raw),w),p)[pick]


def layer():
    prior,update=parent.layer(),correction()
    def apply(raw,p):
        q=prior(raw,p);q[:,:,2]+=update(raw,q)
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    prior,update=ancestor.load(),layer()
    return lambda raw,p:update(raw,prior(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if run():evaluator.outer(sys.modules[__name__])
