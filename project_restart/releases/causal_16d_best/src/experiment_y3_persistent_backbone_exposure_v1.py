"""Confirm the prespecified slow upstream driver at more recent fit cutoffs and against a matched placebo."""
import json
import sys
import numpy as np

import experiment_replicated_expert_consensus_v1 as experts
import experiment_y3_instrumented_market_loadings_v1 as exposure
from experiment_causal_cross_horizon_ridge_v1 import ART
import y3_settled_spread_references as history

parent,base=exposure.parent,exposure.base
EXPERT_IDS=(0,1,3)
OUT=parent.OUT.parent/'y3_persistent_backbone_exposure_v1'
ARMS=('incumbent','same filter with shifted stock weights','fixed upstream slow market')


def common(p):
    return exposure.bands(np.asarray(p).reshape(-1,50,3)).mean(1,keepdims=True)


def historical_heads(day):
    a=np.load(ART/'opening_oof_cache_v1'/f'day_{day:02d}.npy',mmap_mode='r')
    b=np.load(ART/'cutoff_reference_panel_v1'/f'neural_{day}.npy',mmap_mode='r')
    return common(a),common(b)


def apply_market(a,b,w):
    mixture=.65*a+.35*b
    result=[]
    for lo,hi in base.segments(len(a)):
        v=mixture[lo:hi]
        sums=np.concatenate((np.zeros_like(v[:1]),v.cumsum(0)),0)
        end=np.arange(1,len(v)+1)
        left=(end-96).clip(0)
        result.append((sums[end]-sums[left])/(end-left)[:,None,None])
    m=np.concatenate(result)
    out=np.empty((len(m),50),np.float32)
    for j,ids in enumerate(exposure.GROUPS):
        out[:,ids]=.32*(m*w[j]*np.array([.58,.42])).sum(2)
    return out-out.mean(1,keepdims=True)


def train():
    OUT.mkdir(exist_ok=True)
    rows,days=[],[]
    for d in range(4,16):
        if d>=8 and d%2==0:w=exposure.fit(rows)[6]
        _,refs,y=history.training_day(d)
        if d>=8:
            a,b=historical_heads(d)
            z=np.stack((apply_market(a,b,np.roll(w,3,axis=1)),apply_market(a,b,w)),2)
            if d==8:
                n=2431
                np.testing.assert_array_equal(z[:n,:,1],apply_market(a[:n],b[:n],w))
            days.append([base.records(p,y,z) for p in refs])
        moments=[exposure.observations(p,y) for p in refs]
        rows.append(tuple(np.mean(v,0) for v in zip(*moments)))
    def score(group):return [base.score(m,np.ones(2)) for m in np.sum(group,0)]
    halves=np.array([score(days[:4]),score(days[4:])])
    folds=np.array([score(days[i:i+2]) for i in range(0,8,2)])
    daily=np.array([score([d]) for d in days])
    ok=(halves[:,:,:,2].min()>0 and (halves[:,:,0,2]>halves[:,:,0,1]).all()
        and (folds[:,:,0,2]>0).sum(0).min()>=3 and (daily[:,:,0,2]>0).sum(0).min()>=5)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=2 if ok else 0,halves=halves.tolist(),
                                    folds=folds.tolist(),daily=daily.tolist()),indent=2))
    print('selected',bool(ok),'halves',halves[:,:,0].round(6).tolist(),'two-day',folds[:,:,0,2].round(6).tolist(),flush=True)
    if ok:np.savez(OUT/'model.npz',w=exposure.fit(rows)[6])
    return ok


def correction_from_heads():
    with np.load(OUT/'model.npz') as f:
        weights=f['w']
    def apply(opening_predictions,expert_predictions):
        mean_expert=np.mean([common(p) for p in expert_predictions],axis=0)
        return apply_market(common(opening_predictions),mean_expert,weights)
    return apply


def correction():
    import package_opening_expert_v1 as opening
    bundle=ART/'opening_expert_v1/commit_opening_expert'
    ev=opening.Evaluator('',str(bundle),'cuda:0')
    mix=np.array(json.loads((bundle/'conf/Model_0.json').read_text())['opening_mix'])
    nets=experts.sources()
    update=correction_from_heads()
    def apply(raw):
        (_,a),_=opening.replay_features(ev,raw,mix,2)
        return update(a,[nets[i](raw) for i in EXPERT_IDS])
    return apply


def layer():
    before,update=parent.layer(),correction()
    def apply(raw,initial):
        p=before(raw,initial)
        p[:,:,2]+=update(raw)
        return p
    return apply


def replay_with():
    verifier,before=parent.replay_with()
    update=correction()
    def read(d,raw):
        p,h,n=before(d,raw)
        p[:,:,2]+=update(raw)
        h[:,:,2]+=update(raw[:n])
        return p,h,n
    return verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
