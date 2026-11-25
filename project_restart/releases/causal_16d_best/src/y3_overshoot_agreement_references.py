"""Build chronological references after calibrating the common market correction with independent stock baskets."""
import json
import numpy as np
import torch

import experiment_y3_overshoot_basket_agreement_v1 as incumbent
import y3_market_flow_references as previous

market,trial=previous.incumbent,previous.trial
CACHE=incumbent.OUT/'historical_references'


def prepare():
    prior_pick=json.loads((market.OUT/'results.json').read_text())['selected']
    report=json.loads((incumbent.OUT/'results.json').read_text())
    selected=report['selected']
    CACHE.mkdir(exist_ok=True)
    rows=[];errors=[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        z=market.pools(raw,market.prior.teacher())
        base=[market.prior.path.early.load(d,raw)] if d<8 else previous.previous.references(d,raw)
        if d>=8:
            ref=previous.references(d,raw)
            if d%2==0:
                params=dict(market.fit(rows),pick=prior_pick);folds=[[],[]]
            for j,p in enumerate(ref):
                _,signals=incumbent.signal(p,z,params)
                delta=np.broadcast_to(signals[selected][:,None],p.shape[:2])
                stats=trial.scoring.audit.blockstats
                folds[j].append([stats(p,y,[delta]).sum(0),stats(p.mean(1,keepdims=True),y.mean(1,keepdims=True),[delta[:,:1]]).sum(0)])
                p[:,:,2]+=delta
            digest=previous.previous.rolling.cached.digest
            np.savez(CACHE/f'day_{d:02d}.npz',forecasts=ref,cutoff=d-d%2,raw_hash=digest(raw),forecast_hash=digest(ref))
            if d%2:
                values=np.array([[trial.scoring.audit.pooled(g)[0] for g in np.sum(f,axis=0)] for f in folds])
                expected=np.array(report['folds'][(d-9)//2])[:,:,selected]
                errors.append(float(np.max(abs(values-expected))))
                assert errors[-1]<1e-9
        records=[market.record(z,p,y) for p in base]
        rows.append([np.mean(v,axis=0) for v in zip(*records)])
    (CACHE/'verified.json').write_text(json.dumps(dict(fold_errors=errors)))


def references(d,raw):
    return previous.previous.rolling.read_verified(CACHE,d,raw)


if __name__=='__main__':
    torch.set_num_threads(2)
    prepare()
