"""Refit matched controls for repeat-visit features using only the twelve historical training dates."""
import json

import numpy as np
import torch

import experiment_y3_same_quote_memory_v1 as model

study,trial,history=model.study,model.trial,model.history
ARMS=('same price','different price almost same time','latest update','same minus displaced','all same-price revisits')


def design(raw,p,t):
    params=model.parameters()
    sides=params['sides']
    mid,half,_,ok=model.quote.infer(raw,params['k'],params['coefficients'])
    now=np.arange(len(raw))[:,None];stock=np.arange(50)[None,:]
    sets=[[] for _ in range(6)]
    coverage=[]
    for j,level in enumerate((mid-half,mid+half)):
        old=model.previous_visits(level,ok)
        use=(old>=0)&(now-old>1)&(now-old<=240)&ok
        alternate=np.full_like(old,-1)
        # Prefer the immediate neighbor of the matched historical quote, with a deterministic backward-first tie break.
        for h in (1,2,3,4,6):
            for sign in (-1,1):
                ix=old+sign*h
                safe=ix.clip(0,len(raw)-1)
                good=use&(alternate<0)&(ix>=0)&(ix<now)&ok[safe,stock]&(level[safe,stock]!=level)
                alternate[good]=ix[good]
        common=use&(alternate>=0)
        x=np.arcsinh(raw[:,:,sides[j]].astype('float32'))
        diff=np.diff(x,axis=0,prepend=x[:1])
        a=(x-x[old.clip(0),stock])*common[:,:,None]
        b=(x-x[alternate.clip(0),stock])*common[:,:,None]
        coverage.append((int(common.sum()),int(use.sum())))
        for group,v in zip(sets,(diff,a,b,diff*common[:,:,None],a-b,(x-x[old.clip(0),stock])*use[:,:,None])):
            cs=np.concatenate((np.zeros_like(v[:1],dtype=float),v.cumsum(0,dtype=float)))
            end=t+1
            group.extend([(cs[end]-cs[(end-w).clip(0)])/np.sqrt(w) for w in (8,40)])
    z=np.concatenate([np.concatenate(g,axis=2) for g in sets],axis=2)
    return trial.clock.center(np.concatenate((p[t],z),axis=2)).astype('float32'),coverage


def fit(rows):
    indices=[np.arange(19+16*i,35+16*i) for i in range(5)]
    return model.prev.gram(rows,[(ix,19,.35) for ix in indices])


def run():
    rows,folds,coverage=[],[],[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d);y=trial.clock.labels(d).reshape(-1,50,3)
        refs=history.references(d,raw) if d>=8 else [study.screen.history.market.prior.path.early.load(d,raw)]
        if d>=8 and d%2==0:
            state=fit(rows);groups=[[],[]]
        train=np.arange(90+d%21,len(raw)-60,21)
        evaluate=np.arange(d%4,len(raw)-60,4)
        ticks=np.unique(np.r_[train,evaluate]) if d>=8 else train
        rec=[]
        for j,p in enumerate(refs):
            x,c=design(raw,p,ticks)
            if j==0:coverage.append(c)
            rec.append((x[np.searchsorted(ticks,train)],study.error(p,y,train)))
            if d>=8:
                ds=model.changes(x[np.searchsorted(ticks,evaluate)],state)
                g=trial.scoring.audit.blockstats(p[evaluate],y[evaluate],ds).sum(0)
                groups[j].append(g)
        rows.append(tuple(np.concatenate(v) for v in zip(*rec)))
        if d>=8 and d%2:
            folds.append([trial.scoring.audit.pooled(np.sum(g,axis=0)) for g in groups])
    f=np.asarray(folds)
    full=np.array(json.loads((model.OUT/'results.json').read_text())['folds'])[:,:,0,2]
    np.testing.assert_allclose(f[:,:,-1],full,atol=2e-8,rtol=0)
    cnt=np.sum(coverage,axis=(0,1))
    out=dict(arms=ARMS,folds=f.tolist(),mean=f.mean(0).tolist(),eligible=int(cnt[1]),matched=int(cnt[0]))
    (model.OUT/'matched_price_assignment.json').write_text(json.dumps(out,indent=2))
    print(cnt, f.mean(0).round(7), flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
