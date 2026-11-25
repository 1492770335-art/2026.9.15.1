"""Propagate the normalized medium-horizon book correction into its share of the long horizon."""
import json
import numpy as np
import experiment_prefix_book_transport_v1 as parent
import replay_periodic_parent_v1 as replay

clock=parent.clock
OUT=clock.ART/'transport_near_payoff_v1'


def score(rows):
    a,b,y=(np.concatenate(vals).ravel().astype(float) for vals in zip(*rows))
    return float(np.corrcoef(b,y)[0,1]-np.corrcoef(a,y)[0,1])


def screen():
    sd,cov=parent.standard();hist=[];folds=[];daily=[]
    for end in (8,10,12,14):
        hist.extend(parent.record(day,sd,cov) for day in range(len(hist),end))
        w=parent.fit(hist);groups=[[],[]]
        for d in (end,end+1):
            raw=clock.book.rawday(d);t=np.arange(d%3,len(raw),3)
            y=clock.labels(d).reshape(-1,50,3)[t]
            (scale,coupling),_=parent.changes(raw,t,sd,cov,w)
            near=clock.RATIO*scale[:,:,1]
            if d==end:
                ok=t<333
                zs,_=parent.changes(raw[:333],t[ok],sd,cov,w)
                np.testing.assert_allclose(near[ok],clock.RATIO*zs[0][:,:,1],atol=2e-7,rtol=0)
            for g,ref in zip(groups,(clock.incumbent.fi.peer.reference,clock.incumbent.parent.fixed.earlier.early.baseline)):
                p=ref(d).reshape(-1,50,3)[t,:,2]+coupling[:,:,2]
                g.append((p,p+near,y[:,:,2]))
            daily.append([score(g[-1:]) for g in groups])
        folds.append([score(g) for g in groups]);print('window',end+1,folds[-1],flush=True)
    keep=np.min(folds)>0 and (np.array(daily)[:,0]>0).sum()>=5
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,days=daily,keep=bool(keep)),indent=2))
    return keep


def layer():
    base=parent.parent.layer()
    with np.load(parent.OUT/'model.npz') as st:sd,cov,w,keep=(st[k] for k in ('sd','cov','w','keep'))
    assert keep[0,1] and keep[1,2]
    def apply(raw,p):
        q=base(raw,p)
        for t in np.array_split(np.arange(len(q)),min(4,len(q))):
            terms,_=parent.changes(raw,t,sd,cov,w)
            for z,k in zip(terms,keep):q[t]+=z*k
            q[t,:,2]+=clock.RATIO*terms[0][:,:,1]
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as baseline
    upstream=baseline.load()
    update=layer()
    return lambda raw,p: update(raw,upstream(raw,p))


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():replay.verify(OUT,layer,parent.OUT)
