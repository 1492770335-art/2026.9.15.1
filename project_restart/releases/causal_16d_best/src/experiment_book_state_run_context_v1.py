"""Estimate stock-relative returns conditional on the last three distinct categorical book states."""
import numpy as np
import experiment_asymmetric_refresh_transfer_v1 as walk
from experiment_causal_cross_horizon_ridge_v1 import ART,labels

OUT=ART/'book_state_run_context_v1'
PAIRS=((83,90),(85,93),(84,96),(86,98),(89,95),(92,99))
STATES=9
AGES=4
CELLS=STATES**3*AGES


def contexts(raw,pair,events):
    i,j=pair
    z=3*raw[:,:,i].astype(np.int32)+raw[:,:,j].astype(np.int32)
    if z.min()<0 or z.max()>=STATES:raise ValueError('unexpected book state')
    t=np.arange(len(raw))[:,None]
    changed=np.vstack((np.ones_like(z[:1],bool),z[:-1]!=z[1:]))
    start=np.maximum.accumulate(np.where(changed,t,0),axis=0)
    age=np.searchsorted((1,4,13),t-start,side='right')
    cols=np.arange(50)
    if events:
        a=(start-1).clip(0)
        b=(start[a,cols]-1).clip(0)
    else:
        a=(t-4).clip(0);b=(t-9).clip(0)
    return (((z*AGES+age)*STATES+z[a,cols])*STATES+z[b,cols]).astype(np.int32)


def tables(stats):
    count,total=(np.sum(days,axis=0) for days in zip(*stats))
    count=count.reshape(STATES*AGES,STATES,STATES)
    total=total.reshape(STATES*AGES,STATES,STATES,3)
    single=total.sum((1,2))/(count.sum((1,2))[:,None]+60)
    pair=(total.sum(2)+110*single[:,None])/(count.sum(2)[:,:,None]+110)
    triple=(total+200*pair[:,:,None])/(count[:,:,:,None]+200)
    return (triple-single[:,None,None]).reshape(CELLS,3)


def geometry():
    sigma=np.median([walk.book.rawday(d)[:,:,0].astype(float).std(0) for d in range(8)],axis=0).clip(.15)
    all_stats=[[[] for _ in PAIRS] for _ in range(2)]
    for d in range(8):
        raw=walk.book.rawday(d)
        t=np.arange(d%4,len(raw)-60,4)
        y=labels(d).reshape(-1,50,3)[t].astype(float)
        y[:,:,2]-=walk.blend.panel.RATIO*y[:,:,1]
        y[:,:,1]-=(.272/.559)*y[:,:,0]
        y=((y-y.mean(1,keepdims=True))/sigma[None,:,None]).clip(-5,5).reshape(-1,3)
        for k,events in enumerate((False,True)):
            for n,pair in enumerate(PAIRS):
                idx=contexts(raw,pair,events)[t].ravel()
                count=np.bincount(idx,minlength=CELLS)
                total=np.column_stack([np.bincount(idx,weights=y[:,h],minlength=CELLS) for h in range(3)])
                all_stats[k][n].append((count,total))
    lookup=[]
    for kinds in all_stats:
        ps=[]
        for stats in kinds:
            a,b=tables(stats[:4]),tables(stats[4:])
            ps.append(np.sign(a)*np.sqrt(np.abs(a*b))*(a*b>0))
        lookup.append(np.stack(ps))
    return dict(sigma=sigma.astype('float32'),lookup=np.stack(lookup).astype('float32'))


def features(raw,p,geo,fixed_lags):
    k=int(not fixed_lags)
    parts=[]
    for j,pair in enumerate(PAIRS):
        ids=contexts(raw,pair,bool(k))
        parts.append(geo['lookup'][k,j,ids]*geo['sigma'][None,:,None])
    combined=np.mean(parts,axis=0)
    band=p.copy()
    band[:,:,2]-=walk.blend.panel.RATIO*p[:,:,1]
    band[:,:,1]-=(.272/.559)*p[:,:,0]
    parts.extend((combined*np.tanh(abs(band)*3),combined*np.tanh(5*combined*band)))
    z=np.concatenate(parts,axis=2)
    m=z.mean(1)
    return m,z-m[:,None]


if __name__=='__main__':
    walk.run_validation(OUT,geometry(),features,'fixed_lags','distinct_states')
