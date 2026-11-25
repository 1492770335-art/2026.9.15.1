"""Compare otherwise identical peer routes with actual, constant and reassigned depth at price changes."""
import json
import numpy as np
import torch

import experiment_y3_peer_depth_event_causes_v1 as model

source,trial,study,history=model.source,model.trial,model.study,model.history
NAMES=('actual depth','constant volume on moves','other stock volume on moves','lagged depth',
       'permuted graph','improving and worsening prices only','static queue changes only')


def build(raw,p,t,state):
    mid,half,_,_,valid,_,buy,sell=source.queues.tape.prices(raw,state)
    bid,ask=source.queues.size(raw,state)
    prev=np.r_[0,np.arange(len(raw)-1)]
    ok=valid&valid[prev]
    ok[0]=False
    scale=state['volume_scale']
    perm=np.random.default_rng(870223).permutation(50)
    g=source.balanced(state['graphs'][1])
    other=g[np.ix_(perm,perm)]
    stationary=np.zeros_like(bid)
    moves=np.zeros((len(raw),50,2,4))
    for sign,level,size,vol in ((1,mid-half,bid,sell),(-1,mid+half,ask,buy)):
        direction=sign*(level-level[prev])
        stationary+=sign*(direction==0)*(size-size[prev]+vol)
        for j,(mask,idx) in enumerate(((direction>0,np.arange(len(raw))),(direction<0,prev))):
            values=size[idx]/scale
            sgn=sign*(1-2*j)*mask
            for k,q in enumerate((values,np.ones_like(values),np.roll(values,17,axis=1),size[(idx-60).clip(0)]/scale)):
                moves[:,:,j,k]+=sgn*q
    def route(v,graph):
        cs=np.vstack((np.zeros(50),np.where(ok,np.arcsinh(v),0).cumsum(0)))
        return np.stack([source.recent(cs,t,w)@graph for w in (5,20,60)],2)
    fixed=route(stationary/scale,g)
    actual=[route(moves[:,:,j,0],g) for j in (0,1)]
    banks=[(fixed,*actual)]
    for k in (1,2,3):
        banks.append((fixed,*(route(moves[:,:,j,k],g) for j in (0,1))))
    banks.append((route(stationary/scale,other),*(route(moves[:,:,j,0],other) for j in (0,1))))
    banks.extend(((np.zeros_like(fixed),*actual),(fixed,np.zeros_like(fixed),np.zeros_like(fixed))))
    return trial.clock.center(np.concatenate((p[t],*(v for bank in banks for v in bank)),2)).astype('float32')


def run():
    rows=[];folds=[]
    for day in range(4,16):
        raw=trial.clock.book.rawday(day)
        y=trial.clock.labels(day).reshape(-1,50,3)
        refs=history.references(day,raw) if day>=8 else [study.screen.history.market.prior.path.early.load(day,raw)]
        if day>=8 and day%2==0:
            groups=[[],[]]
            gram=source.book.prev.gram(rows,[(np.arange(3+9*i,12+9*i),3,.35) for i in range(len(NAMES))])
        tr=np.arange(90+day%21,len(raw)-60,21)
        val=np.arange(day%4,len(raw)-60,4)
        t=np.unique(np.r_[tr,val]) if day>=8 else tr
        state=source.params(max(4,day-day%2))
        records=[]
        for j,p in enumerate(refs):
            x=build(raw,p,t,state)
            records.append((x[np.searchsorted(t,tr)],study.error(p,y,tr)))
            if day>=8:
                updates=model.changes(x[np.searchsorted(t,val)],gram)
                groups[j].append(trial.scoring.audit.blockstats(p[val],y[val],updates).sum(0))
        rows.append(tuple(np.concatenate(v) for v in zip(*records)))
        if day>=8 and day%2:
            folds.append([trial.scoring.audit.pooled(np.sum(group,axis=0)) for group in groups])
    v=np.asarray(folds)
    original=np.asarray(json.loads((model.OUT/'results.json').read_text())['folds'])[:,:,0,10]
    np.testing.assert_allclose(v[:,:,0],original,atol=4e-8,rtol=0)
    report=dict(days=[9,16],variants=NAMES,mean_y3_gain=v.mean(0).tolist(),folds=v.tolist())
    (model.OUT/'matched_depth_controls.json').write_text(json.dumps(report,indent=2))
    print(np.round(v.mean(0),6),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
