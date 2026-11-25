"""Measure error in orthogonal market, common-factor and stock-specific forecast components."""
from functools import lru_cache
import json
import sys

import numpy as np
import torch

import joint_touch_age_references as history

trial, study = history.trial, history.study
OUT = trial.clock.ART/'y3_risk_subspace_calibration_v1'
ARMS = ('global slope', 'market only','separate factor groups', 'cross-horizon group slopes',
        'relative slopes only', 'reproducible factor corrections', 'permuted stock identities')
GROUPS = (('market',0,1),('next four factors',1,5),('next twelve factors',5,17),
          ('remaining thirty-three factors',17,50))


@lru_cache(maxsize=1)
def geometry():
    grams = []
    for d in range(4):
        y = trial.clock.labels(d).reshape(-1,50,3)
        r = y[60:-60:20,:,1].astype(float)
        r -= r.mean(1,keepdims=True)
        g = np.corrcoef(r.T)
        grams.append(g)
    q = np.ones(50)/np.sqrt(50)
    def basis(g):
        minus = np.eye(50)-np.outer(q,q)
        g = minus@g@minus
        _,v = np.linalg.eigh(g)
        m,_ = np.linalg.qr(np.column_stack((q,v[:,::-1][:,:49])))
        return m
    real = basis(np.mean(grams,0))
    wrong = np.roll(real,17,axis=0)
    return np.stack((real,wrong)).astype('float32')


def moments(p,y,bases):
    a = np.concatenate((p,y[:,:,2:3]),2).astype(float)
    a -= a.mean((0,1),keepdims=True)
    stats = []
    for u in bases:
        coords = np.einsum('tsc,sk->tkc',a,u,optimize=True)
        by_group = []
        for _,l,h in GROUPS:
            v = coords[:,l:h].reshape(-1,4)
            by_group.append((v.T@v)/(len(p)*50))
        stats.append(by_group)
    return np.array(stats)


def slopes(hist):
    gs = np.asarray(hist).mean(1)
    states = []
    for b in range(2):
        cov = gs[:,b]
        total = cov.sum(1).mean(0)
        beta = (total[2,3]/total[2,2]).clip(.4,2.)
        gram = cov[:,:,:3,:3].mean(0)
        rhs = cov[:,:,:3,3]/beta-cov[:,:,:3,2]
        means = rhs.mean(0)
        diag = np.eye(3)[None]*gram.diagonal(axis1=1,axis2=2)[:,None,:].clip(1e-7)
        shared = np.linalg.solve((gram+.4*diag).sum(0),means.sum(0))
        reg = means*abs(means)/(abs(means)+rhs.std(0)/np.sqrt(len(hist))+1e-9)
        w = np.linalg.solve(gram+.75*diag,reg[:,:,None])[:,:,0]
        reliability = abs(np.sign(rhs).mean(0))
        states.append((shared,w,reliability))
    shared,w,agree = states[0]
    single = np.zeros_like(w)
    single[:,2] = w[:,2]
    market = np.zeros_like(w)
    market[0] = w[0]
    local = w.copy()
    local[0] = 0
    return np.stack((np.broadcast_to(shared,w.shape),market,single,w,local,w*agree,states[1][1])).astype('float32')*.35


def changes(p,state):
    w,u = state['weights'],state['basis']
    outputs = []
    for i in range(len(ARMS)):
        basis = u[int(i==6)]
        rotated = np.einsum('tsc,sk->tkc',p,basis,optimize=True)
        coord = np.concatenate([rotated[:,a:b]@weight for (_,a,b),weight in zip(GROUPS,w[i])],axis=1)
        outputs.append(coord@basis.T)
    return outputs


def diagnostics(stats):
    cov = np.asarray(stats).mean((0,1))[0]
    total = cov.sum(0)
    metrics = []
    oracle_covariance = 0.
    for (name,_,_),g in zip(GROUPS,cov):
        w = np.linalg.pinv(g[:3,:3])@g[:3,3]
        oracle_covariance += w@g[:3,3]
        metrics.append(dict(group=name,truth_variance_fraction=g[3,3]/total[3,3],
                            predicted_variance_fraction=g[2,2]/total[2,2],
                            corr=g[2,3]/np.sqrt(g[2,2]*g[3,3])))
    base = total[2,3]/np.sqrt(total[2,2]*total[3,3])
    ceiling = np.sqrt(oracle_covariance/total[3,3])
    return dict(groups=metrics,within_date_centered_corr=base,linear_oracle_corr=ceiling,
                in_sample_room_for_group_recalibration=ceiling-base)


def run():
    OUT.mkdir(exist_ok=True)
    basis = geometry()
    rows,folds,daily = [],[],[]
    for d in range(4,16):
        raw = trial.clock.book.rawday(d)
        y = trial.clock.labels(d).reshape(-1,50,3)
        refs = history.references(d,raw) if d>=8 else [study.screen.history.market.prior.path.early.load(d,raw)]*2
        t = np.arange(d%3,len(y),3)
        if d>=8 and d%2==0:
            state = dict(basis=basis,weights=slopes(rows))
            f = [[],[]]
        rec = []
        for j,p in enumerate(refs):
            rec.append(moments(p[t],y[t],basis))
            if d>=8:
                ds = changes(p[t],state)
                block = trial.scoring.audit.blockstats(p[t],y[t],ds).sum(0)
                f[j].append(block)
                if not j:
                    daily.append(trial.scoring.audit.pooled(block))
        rows.append(rec)
        if d>=8 and d%2:
            folds.append([trial.scoring.audit.pooled(sum(v)) for v in f])
    f = np.array(folds)
    active = []
    for i in (2,3,4,5):
        g = f[:,:,i]
        if (g>0).sum(0).min()>=3 and (g.mean(0)>0).all() and g[-1].min()>0 and (np.array(daily)[:,i]>0).sum()>=5 and all((g-f[:,:,j]).mean(0).min()>0 for j in (0,1,6)):
            active.append(i)
    pick = max(active,key=lambda i:f[:,:,i].mean()) if active else None
    diag = diagnostics(rows[4:])
    report = dict(selected=pick,arms=ARMS,folds=f.tolist(),daily=np.asarray(daily).tolist(),diagnostics=diag)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print('selected',pick,'inner Y3 gains',f.mean(0).round(6),'diagnostics',diag,flush=True)
    if pick is not None:
        np.savez(OUT/'model.npz',basis=basis,weights=slopes(rows),pick=pick)
        history.outer(sys.modules[__name__])


def correction():
    with np.load(OUT/'model.npz') as archive:
        state = dict(archive)
    return lambda raw,p:changes(p,state)[int(state['pick'])]


def layer():
    return study.layer(sys.modules[__name__],history.incumbent)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
