"""Price a pressure spell from its age-dependent survival probability and conditional payoffs."""
import json
import numpy as np
import experiment_peer_catchup_after_pulse_v1 as peer
from experiment_cross_date_response_shapes_v1 import SCALE

OUT=peer.ART/'pressure_spell_survival_v1'
HORIZONS=(5,12,25)
AGE_EDGES=(2,6,15,36,85)
NAGE=len(AGE_EDGES)+1
NCELL=3*2*3
NSIGNAL=3


def trailing(x,n):
    cs=np.concatenate((np.zeros_like(x[:1],dtype=float),x.cumsum(0,dtype=float)))
    end=np.arange(1,len(x)+1)
    start=(end-n).clip(0)
    return ((cs[end]-cs[start])/(end-start).reshape((-1,)+(1,)*(x.ndim-1))).astype('float32')


def spells(raw):
    def side(i,j):
        return np.tanh(np.arcsinh(raw[:,:,i].astype('float32'))-np.arcsinh(raw[:,:,j].astype('float32')))
    flow=np.stack((side(41,42),side(54,55),side(77,78)),axis=2)
    flow=trailing(flow,8)
    prices=trailing(raw[:,:,0].astype('float32')-.00136,16)
    result=[]
    for signal,move,thresh in ((flow,prices,.13),(flow.mean(1,keepdims=True),prices.mean(1,keepdims=True),.035)):
        tick=np.arange(len(raw))[:,None,None]
        strength=abs(signal)/thresh
        strong=np.maximum.accumulate(np.where(strength>=1,tick,-1),axis=0)
        sign=np.where(strong>=0,np.sign(np.take_along_axis(signal,strong.clip(0),axis=0)),0)
        changed=np.concatenate((np.ones_like(sign[:1],bool),sign[1:]!=sign[:-1]))
        since=np.maximum.accumulate(np.where(changed,tick,0),axis=0)
        age=np.digitize(tick-since,AGE_EDGES)
        amplitude=np.digitize(strength,(1,2.5))
        phase=(tick>=400).astype(int)
        aligned=np.digitize(move[:,:,None]*sign,(-.025,.025))
        group=(amplitude*2+phase)*3+aligned
        result.append((sign,since,group,age))
    return result


def bands(y):
    v=y.copy()
    v[:,:,1:]-=y[:,:,:2]*(SCALE[1:]/SCALE[:-1])
    return v


def record(day):
    raw=peer.PARENT.before.book.rawday(day)
    t=np.arange(day%3,len(raw)-60,3)
    y=bands(peer.labels(day).reshape(-1,50,3)[t])
    mean=y.mean(1,keepdims=True)
    tables=np.zeros((2,NSIGNAL,NCELL,NAGE,3,4))
    for s,((direction,start,cell,age),target) in enumerate(zip(spells(raw),(y-mean,mean))):
        for j in range(NSIGNAL):
            sign=direction[t,:,j]
            selected=sign!=0
            state=(cell[t,:,j]*NAGE+age[t,:,j])[selected]
            count=lambda v: np.bincount(state,weights=v,minlength=NCELL*NAGE).reshape(NCELL,NAGE)
            for h,delta in enumerate(HORIZONS):
                survive=(start[t+delta,:,j]==start[t,:,j])[selected]
                ret=(target[:,:,h]*sign)[selected].clip(-5,5)
                tables[s,j,:,:,h]=np.stack((count(None),count(survive),count(survive*ret),count((~survive)*ret)),axis=2)
    return tables


def fit(history):
    n,k,y1,y0=np.moveaxis(np.sum(history,axis=0),-1,0)
    total=n.sum(3,keepdims=True)
    hit=k.sum(3,keepdims=True)
    prior=(hit+1)/(total+2)
    shrink=np.array([160.,25.])[:,None,None,None,None]
    probability=(k+shrink*prior)/(n+shrink)
    premium=y1.sum(3,keepdims=True)/(hit+shrink)-y0.sum(3,keepdims=True)/(total-hit+shrink)
    return prior,probability,((probability-prior)*premium).astype('float32')


def evaluate(raw,t,model,diagnostic=False):
    old,new,effect=model
    sides=[]
    brier=[]
    for s,(direction,start,cell,age) in enumerate(spells(raw)):
        pieces=[]
        per_signal=[]
        for j in range(NSIGNAL):
            c,a,sign=cell[t,:,j],age[t,:,j],direction[t,:,j]
            pieces.append(sign[:,:,None]*effect[s,j,c,a])
            if diagnostic:
                valid=t+max(HORIZONS)<len(raw)
                src=t[valid]
                truth=np.stack([start[src+d,:,j]==start[src,:,j] for d in HORIZONS],axis=2)
                has=sign[valid]!=0
                p1=new[s,j,c[valid],a[valid]]
                p0=old[s,j,c[valid],0]
                per_signal.append(((p0-truth)**2-(p1-truth)**2)[has].mean(0))
        z=.40*np.mean(pieces,axis=0)
        if not s:z-=z.mean(1,keepdims=True)
        sides.append(z)
        if diagnostic:brier.append(np.mean(per_signal,axis=0))
    return sides,brier


def metrics(rows):
    p,y=np.concatenate([r[0] for r in rows]),np.concatenate([r[1] for r in rows])
    b0,b1=bands(p),bands(y)
    parts=[]
    def ic(a,b):
        a=a.reshape(-1,3).astype(float);b=b.reshape(-1,3).astype(float)
        a-=a.mean(0);b-=b.mean(0)
        return (a*b).sum(0)/np.sqrt((a*a).sum(0)*(b*b).sum(0))
    for s in range(2):
        z=np.concatenate([r[2][s] for r in rows])
        a=b0.mean(1,keepdims=True);b=b1.mean(1,keepdims=True)
        if s==0:a,b=b0-a,b1-b
        parts.append(dict(full=(ic(p+z,y)-ic(p,y)).tolist(),band=(ic(a+z,b)-ic(a,b)).tolist()))
    return parts


def screen():
    history,folds,daily,briers=[],[],[],[]
    for end in (8,10,12,14):
        history.extend(record(d) for d in range(len(history),end))
        model=fit(history)
        rows=[]
        for d in (end,end+1):
            raw=peer.PARENT.before.book.rawday(d)
            t=np.arange(d%3,len(raw),3)
            y=peer.labels(d).reshape(-1,50,3)[t]
            p=peer.reference(d)[t]
            sides,brier=evaluate(raw,t,model,True)
            briers.append(np.asarray(brier).tolist())
            if d==end:
                n=221
                head,_=evaluate(raw[:n],t[t<n],model)
                for a,b in zip(head,sides):
                    np.testing.assert_allclose(a,b[:len(a)],rtol=0,atol=2e-6)
            rows.append((p,y,sides))
            daily.append(metrics(rows[-1:]))
        folds.append(metrics(rows))
        print(end+1,folds[-1],flush=True)
    full=np.array([[s['full'] for s in row] for row in folds])
    band=np.array([[s['band'] for s in row] for row in folds])
    days=np.array([[s['full'] for s in row] for row in daily])
    keep=(full.min(0)>0)&(band.mean(0)>0)&((days>0).sum(0)>=5)&(np.mean(briers,axis=0)>0)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,brier_gains=briers,retained=keep.tolist()),indent=2))
    if keep.any():
        history.extend(record(d) for d in range(len(history),16))
        prior,prob,delta=fit(history)
        np.savez(OUT/'model.npz',prior=prior,prob=prob,effect=delta,mask=keep)
    return keep.any()


def load():
    parent=peer.PARENT.load()
    with np.load(OUT/'model.npz') as st:
        model=tuple(st[k] for k in ('prior','prob','effect'))
        mask=st['mask']
    def predict(raw,base):
        q=parent(raw,base).copy()
        sides,_=evaluate(raw,np.arange(len(raw)),model)
        for v,k in zip(sides,mask):
            q+=v*k
        return q
    return predict


def main():
    OUT.mkdir(exist_ok=True)
    if not screen():return
    incumbent=json.loads((peer.PARENT.OUT/'reload_verification.json').read_text())['pooled']['score']
    first=None
    for name in ('raw_replay','reload_verification'):
        result=peer.walk.replay(load())
        if first is not None:
            np.testing.assert_allclose(result['pooled']['candidate'],first['pooled']['candidate'],rtol=0,atol=1e-10)
            for d in result['daily']:
                assert abs(result['daily'][d]['score']-first['daily'][d]['score'])<1e-10
        (OUT/f'{name}.json').write_text(json.dumps(result,indent=2))
        print(name,result['pooled']['score'],flush=True)
        if result['pooled']['score']<=incumbent:break
        first=result


if __name__=='__main__':main()
