"""Validate cross-sectional re-ranking against the retained model using time-ordered out-of-fold scores."""
import json

import numpy as np

import y3_disjoint_validation as trial


def order(score):
    return np.argsort(score,axis=1,kind='stable')


def pair_results(p,y,updates):
    # Compare all 45 pairs inside each set of ten, with half credit for tied predictions.
    i,j=np.triu_indices(10,1)
    t=np.arange(0,len(p),5)
    rows=np.arange(len(t))[:,None]
    ranks=order(p[t,:,2])
    def differences(a):
        a=a[rows,ranks].reshape(-1,5,10)
        return a[:,:,i]-a[:,:,j]
    truth=differences(y[t,:,2])
    keep=truth!=0
    truth=np.sign(truth[keep])
    base=differences(p[t,:,2])[keep]
    hit=lambda diff:np.sum(.5*(1+np.sign(diff)*truth))
    original=hit(base)
    return dict(pairs=int(len(base)),base=float(original),gain=[float(hit(base+differences(u[t])[keep])-original) for u in updates])


def run(module):
    module.OUT.mkdir(exist_ok=True)
    warm=getattr(module,"warm_history",None)
    history=[] if warm is None else warm()
    first=10 if warm is None else 8
    folds=[];daily=[];pairs=[]
    for d in range(8,16):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        z=module.state(raw)
        if d>=first and d%2==0:
            model=module.fit(history)
            stats=[[],[]]
        records=[]
        for j,p in enumerate(trial.refs(d,raw)):
            if d>=first:
                change=module.changes_with_reference(z,model,p)
                if d==first and j==0:
                    for a,b in zip(change,module.changes_with_reference(module.state(raw[:311]),model,p[:311])):
                        np.testing.assert_allclose(a[:311],b,rtol=0,atol=8e-6)
                metric=[v.sum(0) for v in trial.statistics(p,y,change)]
                stats[j].append(metric)
                if not j:
                    daily.append([trial.scoring.audit.pooled(v).tolist() for v in metric])
                    pairs.append(pair_results(p,y,change))
            records.append(module.record(z,p,y,d))
        if hasattr(module, "merge_records"):
            history.append(module.merge_records(records))
        else:
            history.append([np.mean(v,axis=0) for v in zip(*records)])
        if d>=first and d%2:
            folds.append([[trial.scoring.audit.pooled(a).tolist() for a in np.sum(v,axis=0)] for v in stats])
            print('day',d+1,folds[-1][0][0],flush=True)
    pick=module.select(np.asarray(folds),np.asarray(daily),pairs)
    (module.OUT/'results.json').write_text(json.dumps(dict(arms=module.ARMS,selected=pick,folds=folds,daily=daily,pairs=pairs,
               assessed=list(range(first+1,17)),pair_stride=5),indent=2))
    print('selected',pick,flush=True)
    if pick is not None:
        np.savez(module.OUT/'model.npz',pick=pick,**module.fit(history))
        p=trial.previous
        trial.scoring.outer(module,reference=trial.parent,preceding=(p.prior.prior.incumbent,p.prior.incumbent,p.incumbent))


def qualifies(f,d,pairs,a):
    return bool((f[:,:,0,a]>0).all() and (d[:,0,a]>0).sum()>=4 and sum(v['gain'][a] for v in pairs)>0)
