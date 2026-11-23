"""Test carrying the previously selected Y2 increment into the overlapping part of Y3."""
import json
import numpy as np
import experiment_horizon_segment_joint_v1 as incumbent
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, measure

medium = incumbent.medium
online = medium.online
SETTING = medium.CONFIG[0]
RATIO = .160943/.272164
OUT = ART/'verified_near_carry_v1'


def load():
    # The far-market distribution term is added after the new near contribution.
    return incumbent.segment.load(parent=medium.load(carry_medium=RATIO))


def main():
    OUT.mkdir(exist_ok=True)
    stats,weights,folds,daily = [],[],[],[]
    for end in (8,10,12,14):
        prior = medium.prior(stats) if stats else None
        coefficient = medium.fit(weights) if weights else None
        groups = [[],[]]
        for d in (end,end+1):
            for j,path in enumerate((online.best.refit.OUT/f'reference_{d}.npy',online.best.refit.core.OUT/f'pred_{d}.npy')):
                p = np.load(path,mmap_mode='r')
                y = labels(d).reshape(p.shape)
                x,pairs,hist = medium.sample(p,SETTING)
                if j==0:
                    stats.append(hist)
                if prior is None:
                    continue
                z = medium.features(x,pairs,prior,SETTING)
                ticks = np.arange(d%4,len(p),4)
                tr = ticks[ticks<len(p)-60]
                if j==0:
                    weights.append(medium.stats(p[tr,:,1],y[tr,:,1],z[tr]))
                if coefficient is None:
                    continue
                baseline = p[ticks].copy()
                delta = coefficient[1]*z[ticks,:,1]
                baseline[:,:,1] += delta
                candidate = baseline.copy()
                candidate[:,:,2] += RATIO*delta
                group = groups[j]
                group.append(tuple(a.reshape(-1,3) for a in (baseline,candidate,y[ticks])))
                if j==0:
                    daily.append(measure(group[-1:])['gain'])
        if coefficient is not None:
            folds.append([measure(g)['gain'] for g in groups])
            print(end+1,'gain primary/secondary references',folds[-1],flush=True)
    f = np.asarray(folds)
    retained = bool(f[:,0].min()>0 and f[:,1].mean()>0 and (np.array(daily)>0).sum()>=3)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,ratio=RATIO,retained=retained),indent=2))
    if retained:
        best = json.loads((incumbent.OUT/'reload_verification.json').read_text())['pooled']
        for name in ('raw_replay','reload_verification'):
            res=incumbent.segment.best.roll.gls.best.walk.replay(load())
            np.testing.assert_allclose(res['pooled']['candidate'][:2],best['candidate'][:2],rtol=0,atol=1e-10)
            if name=='reload_verification':
                prev=json.loads((OUT/'raw_replay.json').read_text())
                np.testing.assert_allclose(res['pooled']['candidate'],prev['pooled']['candidate'],rtol=0,atol=1e-10)
            (OUT/f'{name}.json').write_text(json.dumps(res,indent=2))
            print(name,res['pooled']['score'],flush=True)
            if res['pooled']['score']<=best['score']:
                break


if __name__=='__main__':
    main()
