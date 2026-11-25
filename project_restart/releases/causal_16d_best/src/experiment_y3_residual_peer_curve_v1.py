"""Calibrate the remaining cross-stock curve signal out of time and diagnose its block uncertainty."""
import json
import numpy as np
import torch

import experiment_y3_predictable_peer_covariance_v1 as cov

curve,clock,incumbent=cov.curve,cov.clock,cov.incumbent
OUT=clock.ART/'y3_residual_peer_curve_v1'


def variables(p,local,pred):
    # Receiver-panel centering matches the prediction target used by the peer graph.
    baseline=np.concatenate((p,local),2).astype('float32')
    for side in (0,1):
        a=baseline[:,side::2]
        a-=a.mean(1,keepdims=True)
    return np.concatenate((baseline,pred[:,:,0,:],pred[:,:,1,0:1]),2)


def record(x,p,y):
    f=x.reshape(-1,x.shape[-1]).astype(float)
    a,b=(clock.center(s[:,:,2]).ravel().astype(float) for s in (p,y))
    beta=((a@b)/(a@a)).clip(.45,2.)
    return f.T@f/len(a),f.T@(b/beta-a)/len(a)


def coefficients(hist):
    g,b=[np.stack(v) for v in zip(*hist)]
    a=g.mean(0)
    nuisance=curve.solve(a[:5,:5],a[:5,5:],.05)
    contr=np.vstack((-nuisance,np.eye(3)))
    residual=contr.T@a@contr
    r=b@contr
    mu=r.mean(0);se=r.std(0)/np.sqrt(len(hist))
    shrink=abs(mu)/(abs(mu)+se+.00001)
    def fit(which):
        w=np.zeros(3)
        w[which]=curve.solve(residual[np.ix_(which,which)],mu[which],.7).clip(-1.5,1.5)*shrink[which]
        return .35*contr@w
    return [fit([0]),fit([0,1]),fit([0,1,2])]


def blockstats(p,y,delta):
    obs=y[:,:,2].astype(float)
    base=p[:,:,2].astype(float)
    rows=[]
    for a in range(0,len(p),120):
        s=slice(a,a+120)
        cols=[base[s],obs[s],*[base[s]+v[s] for v in delta]]
        data=np.stack([c.ravel() for c in cols],axis=1)
        rows.append(np.r_[len(data),data.sum(0),(data.T@data).ravel()])
    return np.stack(rows)


def pooled(stats):
    n=stats.shape[-1]
    dim=int((np.sqrt(4*n-3)-1)/2)
    count=stats[...,0]
    m=stats[...,1:1+dim]/count[...,None]
    gram=stats[...,1+dim:].reshape((*stats.shape[:-1],dim,dim))/count[...,None,None]-m[...,None]*m[...,None,:]
    v=np.diagonal(gram,axis1=-1,axis2=-2)
    corr=gram[...,1,:]/np.sqrt(v*v[...,1,None])
    return corr[...,2:]-corr[...,0,None]


def intervals(days):
    rng=np.random.default_rng(3819)
    experiments=[]
    for _ in range(1600):
        sample=[]
        for which in rng.integers(len(days),size=len(days)):
            blocks=days[which]
            sample.append(blocks[rng.integers(len(blocks),size=len(blocks))].sum(0))
        experiments.append(pooled(np.sum(sample,axis=0)))
    return np.quantile(experiments,[.025,.5,.975],axis=0).tolist()


def signal_sums(p, y, candidates):
    # The last column is withheld Y3; these moments are diagnostic and never enter fitting.
    fields=[np.ones_like(p[:,:,2]),p[:,:,2],*np.moveaxis(candidates,2,0),y[:,:,2]]
    flat=np.column_stack([a.ravel() for a in fields]).astype(float)
    return flat.T@flat


def retrospective_limits(rows):
    # variables() contains five nuisance forecasts and three peer forecasts. Compare
    # nested linear spaces so credit for pre-existing Y1/Y2 is accounted for separately.
    summed=sum(rows)
    covariance=summed[1:,1:]-np.outer(summed[1:,0],summed[1:,0])/summed[0,0]
    def ceiling(cols):
        g=covariance[np.ix_(cols,cols)]
        rhs=covariance[cols,-1]
        return float(np.sqrt((rhs@np.linalg.lstsq(g,rhs,rcond=1e-12)[0])/covariance[-1,-1]))
    original=ceiling([0])
    peers=ceiling([0,6,7,8])
    nuisance=ceiling(list(range(6)))
    joint=ceiling(list(range(9)))
    return dict(rows=int(summed[0,0]), baseline=original, peer_only_ceiling=peers,
                nuisance_ceiling=nuisance, all_ceiling=joint, added_after_nuisance=joint-nuisance)


def run():
    enc=cov.source.encoder(4)
    training,graphs,cal,days,folds,details=[],[],[],[],[],[]
    sums=[[],[]]
    for day in range(16):
        raw=clock.book.rawday(day)
        x=cov.design(raw,enc)
        y=clock.labels(day).reshape(-1,50,3)
        if day>=4:
            if day%2==0:
                base=cov.shared(training)
                if day>=8:
                    projection=cov.maps(graphs)
                    w=coefficients(cal) if cal else None
                    groups=[[],[]]
            local=x@base
            valid=np.arange(len(y)-60)
            if day>=8:
                forecast=cov.moves(local,projection)
                for j,ref in enumerate(curve.REFS):
                    p=ref(day).reshape(y.shape)
                    v=variables(p,local,forecast)
                    sums[j].append(signal_sums(p,y,v))
                    if not j:
                        cal.append(record(v[valid],p[valid],y[valid]))
                    if w is None:
                        continue
                    change=[v@b for b in w]
                    row=blockstats(p,y,change)
                    groups[j].append(row.sum(0))
                    if not j:
                        days.append(row)
                        details.append(pooled(row.sum(0)).tolist())
                        if day%2==0:
                            n=403
                            lc=cov.design(raw[:n],enc)@base
                            small=variables(p[:n],lc,cov.moves(lc,projection))
                            np.testing.assert_allclose(v[:n],small,rtol=0,atol=4e-6)
                if day%2 and w is not None:
                    folds.append([pooled(np.sum(rows,0)).tolist() for rows in groups])
                    print('ending',day+1,folds[-1][0],flush=True)
            graphs.append(cov.stats(local[valid],cov.actual(y,valid)))
        training.append(cov.readout(x,y,day))
    a=np.array(folds)
    ci=intervals(days)
    keep=(a.min((0,1))>0)&((np.array(details)>0).sum(0)>=4)&(np.array(ci[0])>0)
    choices=np.flatnonzero(keep)
    pick=int(max(choices,key=lambda k:a[:,0,k].mean())) if len(choices) else None
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,folds=folds,daily=details,span=120,
        bootstrap_percentiles=ci,pooled_y3_change=pooled(sum(d.sum(0) for d in days)).tolist(),
        retrospective_linear_limits=[retrospective_limits(rows) for rows in sums],
        inner_rows=int(sum(d[:,0].sum() for d in days)),arms=['middle','middle and far','plus realized control']),indent=2))
    if pick is not None:
        np.savez(OUT/'model.npz',mean=enc[0],scale=enc[1],projection=enc[2],read=cov.shared(training),
                 maps=cov.maps(graphs),w=coefficients(cal)[pick])
    return pick is not None


def layer():
    parent=incumbent.layer()
    with np.load(OUT/'model.npz') as s:
        enc=tuple(s[k] for k in ('mean','scale','projection'))
        base,graph,w=s['read'],s['maps'],s['w']
    def apply(raw,p):
        q=parent(raw,p)
        local=cov.design(raw,enc)@base
        q[:,:,2]+=variables(q,local,cov.moves(local,graph))@w
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    parent,update=ancestor.load(),layer()
    return lambda raw,p:update(raw,parent(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    if run():
        cov.source.replay.verify(OUT,layer,incumbent.OUT)
