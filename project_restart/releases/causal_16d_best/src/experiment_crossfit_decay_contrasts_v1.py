"""Cross-fit horizon-specific readouts for quote-state timing and event-clock decay."""
import json
import numpy as np
import experiment_activity_clock_event_decay_v1 as clock

fixed,base,peer=clock.fixed,clock.base,clock.peer
OUT=peer.ART/'crossfit_decay_contrasts_v1'


def basis(raw,t,rate,w):
    # clock.forecast returns dynamic - current and dynamic - event. Rotate into two timing contrasts.
    current,event=clock.forecast(raw,t,rate,w)
    return np.stack((current-event,event),axis=3)


def statistics(p,y,z):
    a,b=(v.reshape(-1,3).astype(float) for v in (p,y))
    x=z.reshape(len(a),3,z.shape[-1]).astype(float)
    a-=a.mean(0);b-=b.mean(0);x-=x.mean(0)
    var=(a*a).mean(0).clip(1.e-5)
    beta=((a*b).mean(0)/var).clip(.35,3)
    x-=a[:,:,None]*((x*a[:,:,None]).mean(0)/var[:,None])
    error=b/beta-a
    x=x.transpose(1,0,2)
    return x.transpose(0,2,1)@x/len(a),(x.transpose(0,2,1)@error.T[:,:,None])[:,:,0]/len(a)


def calibration(history):
    gs,rs=(np.stack(v) for v in zip(*history))
    g=gs.mean(0)
    ridge=1.2*np.eye(2)[None]*g.diagonal(axis1=1,axis2=2).clip(.00001)[:,None,:]
    joint=np.linalg.solve(g+ridge,rs.mean(0)[:,:,None])[:,:,0]
    daily=np.linalg.solve(gs+ridge,rs[:,:,:,None])[:,:,:,0]
    return ((.6*joint+.4*np.median(daily,axis=0)).clip(-.75,.75)*abs(np.sign(daily).mean(0)))


def screen():
    rate=clock.rates()
    models=[];past=[];folds=[];days=[];readouts=[]
    for end in (8,10,12,14):
        models.extend(clock.stats(d,rate) for d in range(len(models),end))
        w=fixed.fit(models)
        coeff=calibration(past) if past else None
        if coeff is not None:readouts.append(coeff.tolist())
        groups=[[],[]]
        for d in (end,end+1):
            raw=base.paths.book.rawday(d);t=np.arange(d%4,len(raw),4)
            z=basis(raw,t,rate,w)
            y=peer.labels(d).reshape(-1,50,3)[t]
            p=peer.reference(d)[t]
            past.append(statistics(p,y,z))
            if coeff is None:continue
            adj=(z*coeff[None,None]).sum(3)
            if d==end:
                st=t[t<211]
                small=basis(raw[:211],st,rate,w)
                np.testing.assert_allclose(z[:len(st)],small,atol=2e-6,rtol=0)
            for rows,ref in zip(groups,(p,fixed.earlier.early.baseline(d).reshape(-1,50,3)[t])):
                rows.append(tuple(a.reshape(-1,3) for a in (ref,ref+adj,y)))
            days.append([peer.delta(r[-1:]) for r in groups])
        if coeff is not None:
            folds.append([peer.delta(r) for r in groups])
            print(end+1,folds[-1],flush=True)
    f=np.array(folds)
    keep=(f[:,0].min(0)>0)&(f[:,1].min(0)>0)&((np.array(days)[:,0]>0).sum(0)>=4)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=days,readouts=readouts,retained=keep.tolist()),indent=2))
    if keep.any():
        models.extend(clock.stats(d,rate) for d in (14,15))
        np.savez(OUT/'model.npz',rate=rate,w=fixed.fit(models),coef=calibration(past)*keep[:,None],mask=keep)
    return keep.any()


def load():
    parent=base.load()
    with np.load(OUT/'model.npz') as st:rate,w,c=(st[k] for k in ('rate','w','coef'))
    def predict(raw,p):
        q=parent(raw,p).copy()
        for k in range(4):
            t=np.arange(k,len(q),4)
            q[t]+=(basis(raw,t,rate,w)*c[None,None]).sum(3)
        return q
    return predict


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():fixed.votes.follow.replay_candidate(OUT,load)
