"""Conditional error of near/far forecasts and their recent revisions across the 20-tick cycle."""
import json
import numpy as np

import experiment_periodic_response_strength_v1 as prior

clock=prior.trial
OUT=clock.ART/'phase_conditioned_forecasts_v1'


def regressors(p,t):
    z=p.astype(float).copy()
    z[:,:,2]-=clock.RATIO*z[:,:,1]
    def design(a):
        now=a[t]
        snap=a[np.maximum(0,t-5)]
        earlier=a[np.maximum(0,t-20)]
        level=.5*(now+earlier)
        return np.concatenate((now,now-snap,snap-earlier,now*np.tanh(4*abs(level))),axis=-1)
    mean=z.mean(1,keepdims=True)
    return design(mean)[:,0],clock.center(design(z-mean))


def fit_record(p,t,y,shift):
    target=y[t].astype(float)
    f=p[t].astype(float)
    # Train each horizon's amplitude against a correlation-calibrated prediction at that horizon.
    a=f-f.mean((0,1));b=target-target.mean((0,1))
    beta=((a*b).mean((0,1))/(a*a).mean((0,1)).clip(1.e-6)).clip(.45,2.5)
    err=target/beta-f
    mean=err.mean(1)
    rows=[]
    for x,truth in zip(regressors(p,t),(mean,err-mean[:,None])):
        by_phase=[]
        for k in range(clock.PERIOD):
            use=clock.phase(t,shift)==k
            v=x[use].reshape(-1,x.shape[-1]);u=truth[use].reshape(-1,3)
            by_phase.append((v.T@v/len(v),v.T@u/len(v)))
        rows.append(tuple(np.stack(c) for c in zip(*by_phase)))
    return rows


def adjustment(p,t,w,shift):
    cycle=clock.phase(t,shift)
    m,r=regressors(p,t)
    return [(.5*np.einsum('ti,tij->tj',m,w[0][cycle]))[:,None],
            .5*clock.center(r@w[1][cycle])]


def screen():
    shift,_=clock.clock_geometry()
    train=[];hist=[];cal=[];folds=[];daily=[]
    for end in (8,10,12,14):
        train.extend(clock.statistics(d,shift) for d in range(len(train),end))
        raw_w=clock.weights(train)
        strength=prior.coefficients(cal) if cal else np.ones(3)
        strength[2]=1
        weight=clock.weights(hist) if hist else None
        groups=[[[],[]],[[],[]]]
        for d in (end,end+1):
            raw=clock.book.rawday(d)
            all_t=np.arange(len(raw))
            # The prior calibration is fitted only on *earlier* rolling windows.
            z=prior.corrections(clock.changes(raw,all_t,shift,raw_w))
            y=clock.labels(d).reshape(-1,50,3)
            t=np.arange(d%3,len(raw),3);tr=t[t<len(raw)-60]
            for j,ref in enumerate((clock.incumbent.fi.peer.reference,clock.incumbent.parent.fixed.earlier.early.baseline)):
                p=ref(d).reshape(y.shape)
                if not j:cal.append(prior.moments(p[tr],y[tr],z[tr]))
                p=(p+z*strength).astype('float32')
                if not j:hist.append(fit_record(p,tr,y,shift))
                if weight is None:continue
                changes=adjustment(p,t,weight,shift)
                for i,g in enumerate(changes):
                    if j==0 and d==end:
                        valid=t<247
                        short=adjustment(p[:247],t[valid],weight,shift)[i]
                        np.testing.assert_allclose(g[valid],short,atol=1e-9,rtol=0)
                    groups[j][i].append((p[t],y[t],np.broadcast_to(g,y[t].shape)))
            if weight is not None:daily.append([clock.deltas(rows[-1:]) for rows in groups[0]])
        if weight is not None:
            folds.append([[clock.deltas(rows) for rows in group] for group in groups])
            print('window',end+1,folds[-1][0],flush=True)
    keep=(np.asarray(folds).min(axis=(0,1))>0)&((np.asarray(daily)>0).sum(0)>=4)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,retained=keep.tolist()),indent=2))
    if keep.any():
        m,r=clock.weights(hist)
        np.savez(OUT/'model.npz',m=m,r=r,shift=shift,mask=keep)
    return keep.any()


def layer():
    parent=prior.layer()
    with np.load(OUT/'model.npz') as st:w=(st['m'],st['r']);mask=st['mask'];shift=int(st['shift'])
    def apply(raw,p):
        q=parent(raw,p)
        for z,k in zip(adjustment(q,np.arange(len(q)),w,shift),mask):q+=z*k
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as baseline
    parent=baseline.load()
    refine=layer()
    return lambda raw,p: refine(raw,parent(raw,p))


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():
        from replay_periodic_parent_v1 import verify
        verify(OUT, layer, prior.OUT)
