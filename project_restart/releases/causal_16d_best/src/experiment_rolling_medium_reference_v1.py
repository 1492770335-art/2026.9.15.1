"""Rebuild the Y2 expert mixer and forecast-transition layers using only prior-date observations."""
import json
import numpy as np
import torch
import experiment_verified_near_carry_v1 as near
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'rolling_medium_reference_v1'
base=near.online.best


def mix(reference,forecasts,models,mask):
    if not models:
        return reference.copy()
    result=reference.copy()
    for start in range(0,len(result),96):
        t=np.arange(start,min(len(result),start+96))
        _,candidate=base.compare(reference,forecasts,t,*models,mask)
        result[t,:,1]=candidate[:,:,1]
    return result


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    roll,expert=base.roll,base.expert
    mask=np.load(expert.OUT/'mask.npy')
    mixture,original,recent=[],[],[]
    history,regress=[],[]
    records=[]
    for end in (8,10,12,14):
        blend=roll.fit(mixture)
        models=[expert.fit(v) for v in (original,recent)] if original else []
        prior=near.medium.prior(history) if history else None
        strength=near.medium.fit(regress) if regress else None
        rows=[]
        for d in (end,end+1):
            fs=roll.sources(d)
            t=np.arange(len(fs[0]))
            y=labels(d).reshape(-1,50,3)
            adaptive=roll.predict(fs,t,blend)
            fixed=np.load(roll.panel.OUT/f'panel_{d}.npy',mmap_mode='r')
            impact=np.load(base.refit.core.OUT/f'pred_{d}.npy',mmap_mode='r')
            old=np.load(base.refit.OUT/f'reference_{d}.npy',mmap_mode='r')
            q=mix(old,fs[1:],models,mask)
            x,obs,hist=near.medium.sample(old,near.SETTING)
            if prior is not None:
                z=near.medium.features(x,obs,prior,near.SETTING)
                ix=t[d%4::4]
                tr=ix[ix<len(t)-60]
                regress.append(near.medium.stats(old[tr,:,1],y[tr,:,1],z[tr]))
            history.append(hist)
            if strength is not None:
                x,obs,_=near.medium.sample(q,near.SETTING)
                z=near.medium.features(x,obs,prior,near.SETTING)
                if d==end:
                    early=mix(old[:173],[s[:173] for s in fs[1:]],models,mask)
                    np.testing.assert_allclose(early,q[:173],atol=2e-6,rtol=0)
                    ex,eo,_=near.medium.sample(early,near.SETTING)
                    np.testing.assert_allclose(near.medium.features(ex,eo,prior,near.SETTING),z[:173],atol=3e-6,rtol=0)
                q[:,:,1]+=strength[1]*z[:,:,1]
            np.save(OUT/f'y2_day_{d}.npy',q[:,:,1].astype('float32'))
            rows.append(tuple(v.reshape(-1,3) for v in (old,q,y)))
            mixture.append(roll.moments(fs,y,t[d%5::5]))
            train=t[d%19::19]
            for history_rows,views in ((original,(adaptive,fixed)),(recent,(impact,old))):
                for p in views:
                    history_rows.append(expert.training_record(p,fs[1:],train,y))
        sc=measure(rows)
        records.append([sc['base'][1],sc['candidate'][1]])
        print(end+1,'full window Y2 correlation old/new reference',records[-1],flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(folds=records,first_day=9,last_day=16),indent=2))


if __name__=='__main__':
    main()
