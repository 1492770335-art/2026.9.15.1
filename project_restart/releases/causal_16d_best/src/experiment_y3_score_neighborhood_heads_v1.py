"""Forecast relative returns inside sets with similar frozen scores while preserving covariance with the original Y3."""
import json
import sys

import numpy as np
import torch

import experiment_y3_settled_price_transfer_v1 as transfer
import experiment_y3_disjoint_forecast_stack_v1 as cache
import experiment_y3_disjoint_peer_gru_v1 as net
import y3_disjoint_validation as trial

OUT=trial.clock.ART/'y3_score_neighborhood_heads_v1'
ARMS=['local final score','local late score','local both','global projection','reassigned groups','raw calibrated',
      'local covariate interactions','half local both']
state=net.state


def residual(features,score,ordering,n_groups):
    order=np.argsort(ordering,axis=1,kind='stable')
    rows=np.arange(len(score))[:,None]
    vals=features[rows,order].reshape(len(score),n_groups,50//n_groups,-1).astype(float)
    vals-=vals.mean(2,keepdims=True)
    q=score[rows,order].reshape(len(score),n_groups,-1).astype(float)
    q-=q.mean(2,keepdims=True)
    denom=np.sum(q*q,axis=2,keepdims=True)
    w=(vals*q[:,:,:,None]).sum(2)/(denom.clip(1e-16))
    vals-=q[:,:,:,None]*w[:,:,None,:]
    output=np.empty(features.shape,dtype='float32')
    output[rows,order]=vals.reshape(features.shape)
    return output


def design(p,h):
    full,late=p[:,:,2],trial.scoring.late(p)
    tilt=late/(.06+late.std(1,keepdims=True))
    z=np.concatenate((h,h*np.tanh(tilt[:,:,None]),h*np.tanh(abs(tilt[:,:,None]))),2)
    due=residual(z,full,full,5)
    far=residual(z,full,late,5)
    global_head=residual(z,full,full,1)
    wrong=residual(z,full,np.roll(full,17,axis=1),5)
    # As a consequence, each projected feature also leaves the pooled covariance with Y3 predictions unchanged.
    np.testing.assert_allclose((due.astype(float)*full[:,:,None]).mean(1),0,atol=3e-8,rtol=0)
    return np.concatenate((due,far,global_head,wrong,h),2)


def fit(hist):
    gs,bs=[np.stack(v) for v in zip(*hist)]
    g=gs.mean(0)
    def solve(ids):
        b=bs[:,ids,0];mu=b.mean(0)
        v=trial.curve.solve(g[np.ix_(ids,ids)],mu*abs(mu)/(abs(mu)+b.std(0)/np.sqrt(len(b))+1e-7),1.0)
        out=np.zeros(g.shape[0]);out[ids]=v.clip(-1.,1.)
        return out
    both=solve([0,1,6,7])
    dirs=[solve([0,1]),solve([6,7]),both,solve([12,13]),solve([18,19]),solve([24,25]),solve(list(range(12))),.5*both]
    return .45*np.column_stack(dirs).astype('float32')


def changes_with_reference(z,m,p):
    h=cache.heads(z,{k[4:]:v for k,v in m.items() if k.startswith('net/')})
    return list(np.moveaxis(design(p,h)@m['w'],2,0))


def select(folds,daily):
    f,d=np.asarray(folds),np.asarray(daily)
    def ok(a,b=None,idx=(0,1)):
        v,s=f[...,a],d[...,a]
        if b is not None:
            v,s=v-f[...,b],s-d[...,b]
        return (v[:,:,0]>0).all() and v[:,:,list(idx)].mean(0).min()>0 and (s[:,0]>0).sum()>=4
    ids=[a for a in [0,1,2,6,7] if ok(a) and ok(a,4,(0,)) and ok(a,5,(0,))]
    return int(max(ids,key=lambda i:f[:,0,0,i].mean())) if ids else None


def run():
    OUT.mkdir(exist_ok=True)
    hist=[];folds=[];daily=[]
    for d in range(8,16):
        raw=trial.clock.book.rawday(d);y=trial.clock.labels(d).reshape(-1,50,3)
        with np.load(cache.CACHE/f'day_{d:02d}.npz') as v:
            assert int(v['train_end'])==d-d%2 and v['raw_hash'].item()==trial.previous.prior.digest(raw)
            h=v['h']
        if d>=10 and d%2==0:
            w=fit(hist);stats=[[],[]]
        records=[]
        for j,p in enumerate(trial.refs(d,raw)):
            x=design(p,h)
            if d>=10:
                m=[a.sum(0) for a in trial.statistics(p,y,list(np.moveaxis(x@w,2,0)))]
                stats[j].append(m)
                if not j:
                    daily.append([trial.scoring.audit.pooled(a).tolist() for a in m])
            records.append(transfer.moments(x,p,y))
        hist.append([np.mean(v,0) for v in zip(*records)])
        if d>=10 and d%2:
            folds.append([[trial.scoring.audit.pooled(a).tolist() for a in np.sum(col,axis=0)] for col in stats])
    pick=select(folds,daily)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=pick,folds=folds,daily=daily),indent=2))
    print('selected',pick,flush=True)
    if pick is not None:
        data=[]
        for d in range(16):
            raw=trial.clock.book.rawday(d)
            data.append(net.observe(net.state(raw),trial.clock.labels(d).reshape(-1,50,3),d))
        model,_=net.fit(data)
        np.savez(OUT/'model.npz',pick=pick,w=fit(hist),**{'net/'+k:v for k,v in model.items()})
        p=trial.previous
        trial.scoring.outer(sys.modules[__name__],reference=trial.parent,preceding=(p.prior.prior.incumbent,p.prior.incumbent,p.incumbent))


def correction():
    return trial.previous.prior.correction(sys.modules[__name__])


def layer():
    return trial.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
