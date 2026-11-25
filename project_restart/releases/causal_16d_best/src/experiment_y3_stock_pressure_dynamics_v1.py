"""Propagate stock-specific book dynamics into five-tick payoffs and test identity against matched substitutes."""
import json
import numpy as np
import torch

import experiment_y3_finite_impact_curve_v1 as impact
import experiment_y3_residual_peer_curve_v1 as audit

curve,clock,incumbent=impact.curve,impact.clock,impact.incumbent
OUT=clock.ART/'y3_stock_pressure_dynamics_v1'


def state(raw):
    x=impact.averages(raw)
    t=np.arange(len(raw))
    v=np.concatenate([x[(t-h).clip(0)] for h in (0,5,10,15)],axis=2)
    return x,v


def stats(raw,y,d):
    x,v=state(raw)
    t=np.arange(15+d%3,len(raw)-5,3)
    a=v[t].transpose(1,0,2).astype(float)
    b=x[t+5].transpose(1,0,2)
    _,coeff,_=curve.settled.historical_geometry()
    r=(y[t,:,0].T-coeff[0,0])*(coeff[2,1]/coeff[0,1])
    r=r.clip(-2,2)
    ma=a.mean(1);mb=b.mean(1)
    a-=ma[:,None];b=b-mb[:,None]
    return ma,a.transpose(0,2,1)@a/len(t),a.transpose(0,2,1)@b/len(t),(a*r[:,:,None]).mean(1)


def systems(hist):
    mu,g,r,pay=[np.mean(v,0) for v in zip(*hist)]
    prior=curve.solve(g.mean(0),r.mean(0),.065)
    read=curve.solve(g.mean(0),pay.mean(0),.22)
    local=[]
    for gram,cross in zip(g,r):
        ridge=np.diag(g.mean(0).diagonal().clip(.003))*.38+.00009*np.eye(len(gram))
        local.append(prior+np.linalg.solve(gram+ridge,cross-gram@prior))
    # Match the wrong identity on book variability; leave current observations and means intact.
    order=np.argsort(g.diagonal(axis1=1,axis2=2).sum(1))
    wrong=np.empty_like(order)
    wrong[order]=order.reshape(5,10)[:,::-1].ravel()
    local=np.stack(local)
    return mu,read,(np.broadcast_to(prior,local.shape),local[wrong],local)


def companion(w):
    transition=np.column_stack((w,np.eye(32)[:,:24]))
    radius=max(abs(np.linalg.eigvals(transition)))
    return transition/min(4.,max(1.,radius/.998))


def fit(hist):
    mu,read,bank=systems(hist)
    def forecast_ops(models):
        values=[];state_maps=[]
        for w in models:
            transition=companion(w)
            power=np.eye(32)
            stream=[];sm=[]
            for j in range(12):
                stream.append(power@read)
                power=power@transition
                if j in (0,6,9):sm.append(power[:,:8])
            stream=np.stack(stream,1)
            values.append(np.column_stack((stream[:,6:10].sum(1),stream[:,4:].sum(1))))
            state_maps.append(sm)
        return np.array(values,dtype='float32'),np.array(state_maps,dtype='float32')
    models=[forecast_ops(s) for s in bank]
    return mu.astype('float32'),np.stack([m[0] for m in models]),np.stack([m[1] for m in models])


def forecast(v,mu,w):
    a=(v-mu).transpose(1,0,2)
    return [np.transpose(a@matrix,(1,0,2)) for matrix in w]


def changes(fs):
    pooled,wrong,matched=fs
    return [.17*clock.center(matched[:,:,0]-pooled[:,:,0]),.17*clock.center(matched[:,:,0]-wrong[:,:,0]),
            .17*clock.center(matched[:,:,1]-wrong[:,:,1])]


def quality(x,v,mu,maps):
    t=np.arange(15,len(x)-50,3)
    start=(v[t]-mu).transpose(1,0,2)
    ratios=[]
    for bank in maps:
        by_h=[]
        for j,h in enumerate((5,35,50)):
            pred=(start@bank[:,j]).transpose(1,0,2)
            truth=x[t+h]-mu[:,:8]
            by_h.append(float(1-np.mean((truth-pred)**2,dtype=float)/np.mean(truth**2,dtype=float)))
        ratios.append(by_h)
    return ratios


def run():
    hist=[];folds=[];daily=[];results={}
    for d in range(16):
        raw=clock.book.rawday(d);y=clock.labels(d).reshape(-1,50,3)
        if d>=8:
            if d%2==0:
                mu,w,sm=fit(hist)
                groups=[[],[]]
            x,v=state(raw);fs=forecast(v,mu,w);z=changes(fs)
            results[d+1]=dict(state_r2=quality(x,v,mu,sm))
            if d%2==0:
                for a,b in zip(z,changes(forecast(state(raw[:349])[1],mu,w))):
                    np.testing.assert_allclose(a[:349],b,rtol=0,atol=2e-6)
            valid=np.arange(len(y)-60);mid=clock.center(impact.blocks(y,valid)[:,:,6:10].sum(2))
            for j,ref in enumerate(curve.REFS):
                p=ref(d).reshape(y.shape)
                if not j:
                    mat=np.column_stack([mid.ravel(),*[clock.center(f[valid,:,0]).ravel() for f in fs]])
                    conf=clock.center(p[valid]).reshape(-1,3).astype(float)
                    res=mat-conf@np.linalg.lstsq(conf,mat,rcond=None)[0]
                    results[d+1]['partial_mid']=np.corrcoef(res.T)[0,1:].tolist()
                s=audit.blockstats(p,y,z).sum(0);groups[j].append(s)
                if not j:daily.append(audit.pooled(s).tolist())
            if d%2:
                folds.append([audit.pooled(np.sum(s,0)).tolist() for s in groups])
                print('ending',d+1,folds[-1][0],flush=True)
        hist.append(stats(raw,y,d))
    a=np.array(folds);r2=np.mean([x['state_r2'] for x in results.values()],axis=0)
    keep=((a>0).sum(0).min(0)>=3)&(a.mean(0).min(0)>0)&(a[-1].min(0)>0)&((np.array(daily)>0).sum(0)>=5)
    choices=np.flatnonzero(keep) if r2[2,1]>max(r2[0,1],r2[1,1]) else []
    pick=int(max(choices,key=lambda k:a[:,0,k].mean())) if len(choices) else None
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,folds=folds,daily=daily,by_day=results,
            states=['pooled','reassigned','matched'],horizons=[5,35,50],arms=['mid minus pooled','mid minus reassigned','far minus reassigned']),indent=2))
    if pick is not None:
        mu,w,_=fit(hist);np.savez(OUT/'model.npz',mu=mu,w=w,pick=pick)
    return pick is not None


def layer():
    parent=incumbent.layer()
    with np.load(OUT/'model.npz') as s:mu,w,pick=s['mu'],s['w'],int(s['pick'])
    def apply(raw,p):
        q=parent(raw,p)
        q[:,:,2]+=changes(forecast(state(raw)[1],mu,w))[pick]
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    parent,update=ancestor.load(),layer()
    return lambda raw,p:update(raw,parent(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if run():impact.source.replay.verify(OUT,layer,incumbent.OUT)
