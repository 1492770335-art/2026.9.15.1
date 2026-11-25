"""Test whether the distance between stock forecasts carries more Y3 information than their within-tick ranks."""
import json
import sys

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata
import torch

import y3_pair_incumbent_validation as trial

OUT = trial.clock.ART/'y3_rank_spacing_calibration_v1'
ARMS = ['linear calibration', 'Y3 rank spacing', 'short rank spacing', 'all three horizons',
        'cross-horizon shape', 'reassigned short ranks', 'half Y3 spacing', 'half joint spacing']


def state(raw):
    return None


def features(p):
    q = p.astype('float32')
    relative = q-q.mean(1, keepdims=True)
    ranks = rankdata(q, axis=1, method='average')
    normal = ndtri((ranks-.5)/50).astype('float32')
    scale = np.sqrt(np.mean(relative**2, axis=1, keepdims=True)).clip(.01)
    # An exactly Gaussian cross section has little spacing correction. Pure scale changes cannot change this contrast.
    distance = normal*scale-relative
    outer = np.abs(normal)>1.0
    wing = distance*outer
    skew = scale*(normal**2-1)
    cross = distance*np.tanh(normal[:,:,2:3]*normal[:,:,:1])
    design = np.concatenate((relative, distance, wing, skew, cross, np.roll(distance[:,:,:2],17,axis=1)), axis=2)
    return trial.clock.center(design).astype('float32')


def record(p,y,d):
    t = np.arange(d%4, len(p), 4)
    x = features(p[t]).reshape(-1,17).astype(float)
    pred = p[t,:,2].ravel().astype(float)
    truth = y[t,:,2].ravel().astype(float)
    beta = np.cov(pred,truth,bias=True)[0,1]/max(1e-8,pred.var())
    error = (truth/np.clip(beta,.35,2.5)-pred)
    return x.T@x/len(x), x.T@error/len(x)


def fit(history):
    grams,rhs = [np.stack(items) for items in zip(*history)]
    g = grams.mean(0)
    def solve(cols):
        mu = rhs[:,cols].mean(0)
        mu *= abs(mu)/(abs(mu)+rhs[:,cols].std(0)/len(rhs)**.5+1e-7)
        w = np.zeros(len(g))
        w[cols] = trial.curve.solve(g[np.ix_(cols,cols)],mu,.6)
        return w
    linear = solve([0,1,2])
    own = solve([5,8,11])
    short = solve([3,4,6,7,9,10])
    joint = solve(list(range(3,12)))
    interaction = solve(list(range(3,15)))
    wrong = solve([5,8,11,15,16])
    return dict(w=.45*np.stack((linear,own,short,joint,interaction,wrong,.5*own,.5*joint),1).astype('float32'))


def changes_with_reference(z,m,p):
    output = np.concatenate([features(q)@m['w'] for q in np.array_split(p,(len(p)+127)//128)])
    return list(np.moveaxis(output,2,0))


def select(f,d):
    eligible = []
    for a in (1,2,3,4,6,7):
        v = f[:,:,0,a]
        advantage = v-f[:,:,0,0]
        if ((v>0).all() and (d[:,0,a]>0).sum()>=4 and advantage.mean(0).min()>0
            and (a not in (2,3,4) or (v-f[:,:,0,5]).mean(0).min()>0)):
            eligible.append(a)
    return int(max(eligible,key=lambda i:f[:,0,0,i].mean())) if eligible else None


def run(module=None, references=trial.refs, outer=None):
    mod = module or sys.modules[__name__]
    mod.OUT.mkdir(exist_ok=True)
    hist,folds,daily = [],[],[]
    warm = getattr(mod, "warm_history", None)
    if warm is not None:
        hist.extend(warm())
    first = 8 if hist else 10
    for d in range(8,16):
        raw = trial.clock.book.rawday(d)
        y = trial.clock.labels(d).reshape(-1,50,3)
        historical_state = getattr(mod, "historical_state", None)
        z = historical_state(raw,d) if historical_state else mod.state(raw)
        if d>=first and d%2==0:
            m = mod.fit(hist)
            groups = [[],[]]
        records = []
        for j,p in enumerate(references(d,raw)):
            if d>=first:
                change = mod.changes_with_reference(z,m,p)
                if d==first and not j:
                    short = historical_state(raw[:311],d) if historical_state else mod.state(raw[:311])
                    prefix = mod.changes_with_reference(short,m,p[:311])
                    for a,b in zip(change,prefix):
                        np.testing.assert_allclose(a[:311],b,rtol=0,atol=4e-6)
                stats = [v.sum(0) for v in trial.previous.statistics(p,y,change)]
                groups[j].append(stats)
                if not j:
                    daily.append([trial.scoring.audit.pooled(s).tolist() for s in stats])
            records.append(mod.record_with_state(z,p,y,d) if hasattr(mod,'record_with_state') else mod.record(p,y,d))
        merge = getattr(mod, "merge_records", None)
        hist.append(merge(records) if merge else [np.mean(a,0) for a in zip(*records)])
        if d>=first and d%2:
            folds.append([[trial.scoring.audit.pooled(s).tolist() for s in np.sum(g,0)] for g in groups])
    chosen = mod.select(np.array(folds),np.array(daily))
    result = dict(selected=chosen,arms=mod.ARMS,folds=folds,daily=daily,dates=list(range(first+1,17)))
    (mod.OUT/'results.json').write_text(json.dumps(result,indent=2))
    print('selected',chosen,flush=True)
    if chosen is not None:
        np.savez(mod.OUT/'model.npz',pick=chosen,**mod.fit(hist))
        if outer is not None:
            outer(mod)
            return
        old = trial.previous.previous
        trial.scoring.outer(mod,reference=trial.incumbent,preceding=(old.prior.prior.incumbent,old.prior.incumbent,
                                                                    old.incumbent,trial.previous.parent))


def correction():
    return trial.previous.previous.prior.correction(sys.modules[__name__])


def layer():
    return trial.layer(sys.modules[__name__])


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
