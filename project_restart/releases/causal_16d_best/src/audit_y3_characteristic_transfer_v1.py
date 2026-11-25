"""Decompose verified stock-specific adjustments by stock, session, and return horizon."""
import json
import numpy as np

import experiment_y3_realized_sort_rotation_v1 as rotation
from audit_y3_session_market_band_attribution_v1 import summarize

portfolios,clock,parent=rotation.portfolios,rotation.clock,rotation.parent


def moments(pred,target,delta):
    x=np.stack((np.ones_like(pred),pred,target,delta),axis=2).transpose(1,0,2).astype(float)
    return x.transpose(0,2,1)@x


def run():
    verifier,read=parent.parent.replay_with()
    previous=parent.correction_from_heads();common=clock.correction()
    with np.load(portfolios.OUT/'model.npz') as f:pw=dict(f)
    with np.load(rotation.OUT/'model.npz') as f:rw=dict(f)
    geo=float(rw['k']),rw['coef'],rw['row']
    data=[];peak=0.
    for d in range(16,20):
        raw=verifier.trace.replay.original.features(verifier.trace.replay.original.DATA,d)
        p,_,_=read(d,raw)
        opening=np.load(parent.ART/'opening_oof_cache_v1'/f'day_{d:02d}.npy').reshape(p.shape)
        nets=[np.load(parent.ART/parent.experts.experts.SOURCES[i]/f'side_{d}.npy').reshape(p.shape)
              for i in parent.EXPERT_IDS]
        p[:,:,2]+=previous(opening,nets)
        correction=portfolios.corrections(raw,p,pw)[:,:,int(pw['pick'])-1]+rotation.signals(raw,p,rw,geo)[:,:,int(rw['pick'])-1]
        p[:,:,2]+=common(raw,p)
        y=verifier.trace.replay.labels(d).reshape(p.shape)
        peak=max(peak,abs(correction.mean(1,dtype=float)).max())
        day=[]
        for lo,hi in parent.base.segments(len(raw)):
            pair=[]
            for scale in (0.,parent.base.RATIO):
                fore=p[lo:hi,:,2]-scale*p[lo:hi,:,1]
                target=y[lo:hi,:,2]-scale*y[lo:hi,:,1]
                delta=correction[lo:hi]
                full=moments(fore,target,delta)
                rel=moments(fore-fore.mean(1,keepdims=True),target-target.mean(1,keepdims=True),
                            delta-delta.mean(1,keepdims=True))
                pair.append(np.stack((full,rel)))
            day.append(pair)
        data.append(day)
    data=np.array(data)
    pooled=data.sum((0,1))
    summary=[[summarize(m.sum(0)) for m in row] for row in pooled]
    total=pooled[0,0].sum(0)
    new=json.loads((rotation.OUT/'cached_validation.json').read_text())['pooled']
    old=json.loads((clock.OUT/'cached_validation.json').read_text())['pooled']
    assert abs(.5*summary[0][0]['change']-(new['score']-old['score']))<2e-8
    assert peak<1e-6
    omitted=[.5*summarize(total-v)['change'] for v in pooled[0,0]]
    per_stock=[summarize(v)['change'] for v in pooled[0,0]]
    per_session=[.5*summarize(v[0,0].sum(0))['change'] for v in data.sum(0)]
    result=dict(rows=int(total[0,0]),comparison='both portfolio layers relative to the common-market clock model',
                y3=dict(pooled=summary[0][0],cross_section=summary[0][1]),
                late=dict(pooled=summary[1][0],cross_section=summary[1][1]),
                max_abs_market_update=float(peak),positive_individual_stock_ic=int((np.array(per_stock)>0).sum()),
                individual_stock_ic_gain=per_stock,
                omit_stock_pooled_gain_range=[min(omitted),max(omitted)],
                per_session_weighted_gain=per_session)
    (rotation.OUT/'stock_and_horizon_attribution.json').write_text(json.dumps(result,indent=2))
    print('market max',peak,'positive stocks',result['positive_individual_stock_ic'],
          'leave stock out',result['omit_stock_pooled_gain_range'],'late',summary[1],flush=True)


if __name__=='__main__':run()
