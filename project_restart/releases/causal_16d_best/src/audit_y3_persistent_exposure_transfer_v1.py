"""Compare frozen upstream predictions against replay, and bootstrap the four paired holdout dates."""
import json
import numpy as np
import experiment_y3_persistent_backbone_exposure_v1 as model

OUT=model.OUT
ART=model.ART


def statistics(p,y,z):
    v=np.stack((np.ones_like(z),p[:,:,2],y[:,:,2],z),2).reshape(-1,4).astype(float)
    return v.T@v


def gain(stats):
    c=stats[...,1:,1:]-stats[...,1:,0,None]*stats[...,0,None,1:]/stats[...,0,0,None,None]
    r0=c[...,0,1]/np.sqrt(c[...,0,0]*c[...,1,1])
    r1=(c[...,0,1]+c[...,2,1])/np.sqrt((c[...,0,0]+2*c[...,0,2]+c[...,2,2])*c[...,1,1])
    return .5*(r1-r0)


def run():
    correction=model.correction_from_heads()
    groups=[]
    verified=[]
    for d in range(16,20):
        v,read=model.parent.replay_with()
        raw=v.trace.replay.original.features(v.trace.replay.original.DATA,d)
        p,_,_=read(d,raw)
        y=v.trace.replay.labels(d).reshape(p.shape)
        a=np.load(ART/'opening_oof_cache_v1'/f'day_{d:02d}.npy').reshape(p.shape)
        nets=[np.load(ART/model.experts.experts.SOURCES[i]/f'side_{d}.npy').reshape(p.shape)
              for i in model.EXPERT_IDS]
        z=correction(a,nets)
        daily=[]
        for lo,hi in model.base.segments(len(p)):
            for t in range(lo,hi,120):
                sl=slice(t,min(t+120,hi))
                daily.append(statistics(p[sl],y[sl],z[sl]))
        daily=np.stack(daily)
        approx=float(gain(daily.sum(0)))
        verified.append(dict(day=d+1,cached_score_gain=approx))
        groups.append(daily)
    rng=np.random.default_rng(716453)
    draws=[]
    for _ in range(2500):
        summed=np.zeros((4,4))
        for j in rng.integers(0,4,4):
            g=groups[j]
            summed+=g[rng.integers(0,len(g),len(g))].sum(0)
        draws.append(gain(summed))
    info=dict(day_checks=verified,blocks_in_ticks=120,replicates=len(draws),gain_interval=np.quantile(draws,[.025,.975]).tolist(),
              gain_from_cache=float(gain(np.sum([g.sum(0) for g in groups],0))))
    path=OUT/'raw_replay.json'
    if not path.is_file():
        print('cached inputs only',info,flush=True)
        return
    report=json.loads(path.read_text())
    for row in verified:
        row['difference_from_raw_replay']=row['cached_score_gain']-report['daily'][str(row['day'])]['gain']
    # A stale upstream cache cannot certify the live candidate.
    assert max(abs(r['difference_from_raw_replay']) for r in verified)<1e-6,verified
    (OUT/'paired_uncertainty.json').write_text(json.dumps(info,indent=2))
    print(info,flush=True)


if __name__=='__main__':run()
