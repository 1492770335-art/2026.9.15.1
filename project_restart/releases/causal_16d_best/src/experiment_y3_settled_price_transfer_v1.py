"""Learn the signed transfer from settled same-panel and other-panel slopes to the next forecast."""
import json
import sys

import numpy as np
import torch

import experiment_y3_settled_direct_heads_v1 as live
import experiment_y3_disjoint_forecast_stack_v1 as historical
import experiment_y3_disjoint_peer_gru_v1 as net
import y3_disjoint_validation as trial

OUT=trial.clock.ART/'y3_settled_price_transfer_v1'
ARMS=['shared coefficient','panel difference','both coefficients','both plus direct heads',
      'reassigned panels','delayed settlements','direct heads','both plus tail objective','half both']
SECONDARY=live.SECONDARY
state=live.state


def parts(h,p,settled):
    values=[]
    for shift,delay in [(0,0),(16,0),(0,80)]:
        predictions=live.calibration(h,p,settled,shift,delay)
        other,own=predictions[:,:,0],predictions[:,:,1]
        values.extend([.5*(own+other),.5*(own-other)])
    return np.stack([*values,h[:,:,0],h[:,:,1]],axis=2)


def moments(x,p,y):
    z=x.reshape(-1,x.shape[-1]).astype(float)
    def residual(p,y):
        v=p.ravel().astype(float);u=y.ravel().astype(float)
        v-=v.mean();u-=u.mean()
        scale=(v@u)/max(1e-8,v@v)
        return u/scale.clip(.25,2.5)-v
    tail=trial.scoring.late
    residuals=np.stack((residual(p[:,:,2],y[:,:,2]),residual(tail(p),tail(y))),1)
    return z.T@z/len(z),z.T@residuals/len(z)


def fit(history):
    gs,bs=[np.stack(v) for v in zip(*history)]
    g=gs.mean(0)
    def solve(ids,include_tail=False):
        cross=bs[:,ids,:]
        v=cross.mean(0)
        v*=abs(v)/(abs(v)+cross.std(0)/np.sqrt(len(history))+1e-6)
        w=trial.curve.solve(g[np.ix_(ids,ids)],v,1.2)
        w=.75*w[:,0]+.25*w[:,1] if include_tail else w[:,0]
        out=np.zeros(8);out[ids]=w.clip(-1.2,1.2)
        return out
    joint=solve([0,1])
    weights=[solve([0]),solve([1]),joint,solve([0,1,6,7]),solve([2,3]),solve([4,5]),
             solve([6,7]),solve([0,1],True),.5*joint]
    return .65*np.stack(weights,axis=1).astype('float32')


def changes_with_reference(s,m,p):
    z,settled=s
    h=historical.heads(z,{k[4:]:v for k,v in m.items() if k.startswith('net/')})
    return list(np.moveaxis(parts(h,p,settled)@m['w'],2,0))


def select(folds,daily):
    f,d=np.asarray(folds),np.asarray(daily)
    def good(a,b=None,indices=(0,1)):
        v,h=f[...,a],d[...,a]
        if b is not None:
            v,h=v-f[...,b],h-d[...,b]
        q=v[:,:,list(indices)]
        return (q[:,:,:1]>0).all() and q.mean(0).min()>0 and (h[:,0]>0).sum()>=4
    choices=[a for a in (0,1,2,3,7,8) if good(a) and all(good(a,b,(0,)) for b in (4,5))]
    return int(max(choices,key=lambda i:f[:,0,0,i].mean())) if choices else None


def run():
    OUT.mkdir(exist_ok=True)
    hist=[];folds=[];daily=[];fits=[]
    for d in range(8,16):
        raw=trial.clock.book.rawday(d);y=trial.clock.labels(d).reshape(-1,50,3)
        settled=live.settlement(raw)
        with np.load(historical.CACHE/f'day_{d:02d}.npz') as v:
            assert v['raw_hash'].item()==trial.previous.prior.digest(raw) and int(v['train_end'])==d-d%2
            h=v['h']
        if d>=10 and d%2==0:
            coeff=fit(hist);fits.append(coeff.tolist());stats=[[],[]]
        records=[]
        for j,p in enumerate(trial.refs(d,raw)):
            x=parts(h,p,settled)
            if d>=10:
                delta=list(np.moveaxis(x@coeff,2,0))
                m=[b.sum(0) for b in trial.statistics(p,y,delta)]
                stats[j].append(m)
                if not j:
                    daily.append([trial.scoring.audit.pooled(b).tolist() for b in m])
            records.append(moments(x,p,y))
        hist.append([np.mean(v,0) for v in zip(*records)])
        if d>=10 and d%2:
            folds.append([[trial.scoring.audit.pooled(q).tolist() for q in np.sum(v,axis=0)] for v in stats])
    pick=select(folds,daily)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=pick,folds=folds,daily=daily,
                                                    weights=fits,assessment_days=list(range(11,17))),indent=2))
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
