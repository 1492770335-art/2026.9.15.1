"""Measure whether the accepted common correction carries information about days 17–20 beyond the first 20 ticks."""
import json
import numpy as np
import experiment_y3_session_continuation_ridge_v1 as clock

parent=clock.parent


def gram(x,y,z):
    v=np.column_stack((np.ones(x.size),x.ravel(),y.ravel(),z.ravel())).astype(float)
    return v.T@v


def summarize(m):
    v=m[1:,1:]-np.outer(m[1:,0],m[1:,0])/m[0,0]
    old=v[0,1]/np.sqrt(v[0,0]*v[1,1])
    new=(v[0,1]+v[2,1])/np.sqrt((v[0,0]+2*v[0,2]+v[2,2])*v[1,1])
    return dict(before=float(old),after=float(new),change=float(new-old))


def run():
    verifier,read=parent.parent.replay_with()
    market=parent.correction_from_heads()
    adjust=clock.correction()
    matrices=[];daily=[]
    for d in range(16,20):
        raw=verifier.trace.replay.original.features(verifier.trace.replay.original.DATA,d)
        p,_,_=read(d,raw)
        a=np.load(parent.ART/'opening_oof_cache_v1'/f'day_{d:02d}.npy').reshape(p.shape)
        heads=[np.load(parent.ART/parent.experts.experts.SOURCES[i]/f'side_{d}.npy').reshape(p.shape) for i in parent.EXPERT_IDS]
        p[:,:,2]+=market(a,heads)
        y=verifier.trace.replay.labels(d).reshape(p.shape)
        z=adjust(raw,p)
        day=[]
        for h in (0,1):
            target=y[:,:,2]-h*parent.base.RATIO*y[:,:,1]
            forecast=p[:,:,2]-h*parent.base.RATIO*p[:,:,1]
            day.append(gram(forecast,target,z))
            day.append(gram(forecast.mean(1),target.mean(1),z[:,0]))
        matrices.append(day)
        daily.append([summarize(m) for m in day])
    pooled=[summarize(m) for m in np.sum(matrices,0)]
    expected=json.loads((clock.OUT/'reload_verification.json').read_text())['pooled']['gain']
    assert abs(.5*pooled[0]['change']-expected)<2e-8
    result=dict(scopes=['full_y3','full_y3_market_mean','late_interval','late_interval_market_mean'],
                pooled=pooled,daily=daily,rows=int(np.sum(matrices,0)[0,0,0]))
    (clock.OUT/'band_attribution.json').write_text(json.dumps(result,indent=2))
    print('pooled changes',[m['change'] for m in pooled],flush=True)


if __name__=='__main__':run()
