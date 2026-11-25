"""Calculate paired trading-day and intraday block uncertainty for the fixed market correction."""
import json
import numpy as np
import experiment_y3_session_continuation_ridge_v1 as model
from audit_y3_persistent_exposure_transfer_v1 import statistics,gain

parent=model.parent


def run(use_portfolios=False, rotation=False, from_common=False):
    verifier,read=parent.parent.replay_with()
    upstream=parent.correction_from_heads()
    apply=model.correction()
    out=model.OUT
    if use_portfolios or rotation:
        import experiment_y3_session_characteristic_portfolios_v1 as portfolios
        with np.load(portfolios.OUT/'model.npz') as data:
            state=dict(data)
        out=portfolios.OUT
    if rotation:
        import experiment_y3_realized_sort_rotation_v1 as settled
        with np.load(settled.OUT/'model.npz') as data:
            rotated=dict(data)
        geo=float(rotated['k']),rotated['coef'],rotated['row']
        out=settled.OUT
    groups=[]
    for day in range(16,20):
        raw=verifier.trace.replay.original.features(verifier.trace.replay.original.DATA,day)
        p,_,_=read(day,raw)
        opening=np.load(parent.ART/'opening_oof_cache_v1'/f'day_{day:02d}.npy').reshape(p.shape)
        heads=[np.load(parent.ART/parent.experts.experts.SOURCES[k]/f'side_{day}.npy').reshape(p.shape)
               for k in parent.EXPERT_IDS]
        p[:,:,2]+=upstream(opening,heads)
        y=verifier.trace.replay.labels(day).reshape(p.shape)
        common=apply(raw,p)
        if use_portfolios or rotation:
            z=portfolios.corrections(raw,p,state)[:,:,int(state['pick'])-1]
            if rotation:
                update=settled.signals(raw,p,rotated,geo)[:,:,int(rotated['pick'])-1]
                if from_common:
                    z=z+update
                else:
                    common=common+z
                    z=update
            p[:,:,2]+=common
        else:
            z=common
        day_blocks=[]
        for lo,hi in parent.base.segments(len(raw)):
            for start in range(lo,hi,120):
                sl=slice(start,min(start+120,hi))
                day_blocks.append(statistics(p[sl],y[sl],z[sl]))
        groups.append(np.stack(day_blocks))
    rng=np.random.default_rng(82713)
    sums=np.stack([v.sum(0) for v in groups])
    draws=[]
    for _ in range(2500):
        sample=np.zeros_like(sums[0])
        for day in rng.integers(4,size=4):
            blocks=groups[day]
            sample+=blocks[rng.integers(len(blocks),size=len(blocks))].sum(0)
        draws.append(float(gain(sample)))
    pooled=float(gain(sums.sum(0)))
    report=dict(rows=int(sums[:,0,0].sum()),pooled_gain=pooled,block_width=120,replicates=len(draws),
                omit_date={day+17:float(gain(sums.sum(0)-v)) for day,v in enumerate(sums)},
                paired_interval=np.quantile(draws,[.025,.975]).tolist())
    expected=json.loads((out/'cached_validation.json').read_text())['pooled']
    baseline_dir=portfolios.OUT if rotation and not from_common else model.OUT
    base=(json.loads((baseline_dir/'cached_validation.json').read_text())['pooled']['score']
          if use_portfolios or rotation else expected['baseline_score'])
    assert abs(expected['score']-base-pooled)<1e-8
    path='paired_from_common.json' if rotation and from_common else 'paired_uncertainty.json'
    (out/path).write_text(json.dumps(report,indent=2))
    print(report,flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--characteristic-portfolios',action='store_true')
    parser.add_argument('--rotation',action='store_true')
    parser.add_argument('--from-common',action='store_true')
    args=parser.parse_args()
    run(args.characteristic_portfolios,args.rotation,args.from_common)
