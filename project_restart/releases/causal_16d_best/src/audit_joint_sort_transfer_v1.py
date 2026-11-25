"""Pair both target improvements on the same trading days and the same intraday blocks."""
import json
import numpy as np

import experiment_joint_sort_horizon_transfer_v1 as joint

clock,parent,rotation=joint.clock,joint.parent,joint.rotation


def gain(matrix):
    n=matrix[:,0,0,None,None]
    cov=matrix[:,1:,1:]-matrix[:,1:,0,None]*matrix[:,0,None,1:]/n
    old=cov[:,0,1]/np.sqrt(cov[:,0,0]*cov[:,1,1])
    new=(cov[:,0,1]+cov[:,2,1])/np.sqrt((cov[:,0,0]+2*cov[:,0,2]+cov[:,2,2])*cov[:,1,1])
    parts=(new-old)*[.3,.5]
    return np.r_[parts,parts.sum()]


def run():
    verifier,read=parent.parent.replay_with()
    upstream=parent.correction_from_heads()
    common=clock.correction(); y2=joint.correction();y3=rotation.correction()
    groups=[]
    for d in range(16,20):
        raw=verifier.trace.replay.original.features(verifier.trace.replay.original.DATA,d)
        p,_,_=read(d,raw)
        opening=np.load(parent.ART/'opening_oof_cache_v1'/f'day_{d:02d}.npy').reshape(p.shape)
        heads=[np.load(parent.ART/parent.experts.experts.SOURCES[i]/f'side_{d}.npy').reshape(p.shape)
               for i in parent.EXPERT_IDS]
        p[:,:,2]+=upstream(opening,heads)
        target=verifier.trace.replay.labels(d).reshape(p.shape)[:,:,1:]
        old=p[:,:,1:].copy();new=old.copy()
        old[:,:,1]+=common(raw,p)
        new[:,:,1]+=y3(raw,p)
        new[:,:,0]+=y2(raw,p)
        blocks=[]
        for lo,hi in parent.base.segments(len(p)):
            for start in range(lo,hi,120):
                sl=slice(start,min(start+120,hi))
                a=old[sl].reshape(-1,2).T.astype(float)
                b=target[sl].reshape(-1,2).T.astype(float)
                z=new[sl].reshape(-1,2).T.astype(float)-a
                values=np.stack((np.ones_like(a),a,b,z),2)
                blocks.append(values.transpose(0,2,1)@values)
        groups.append(np.stack(blocks))
    sums=np.stack([v.sum(0) for v in groups]);total=sums.sum(0)
    daily=[gain(s).tolist() for s in sums];pooled=gain(total)
    orig=json.loads((clock.OUT/'cached_validation.json').read_text())['pooled']
    curr=json.loads((joint.OUT/'cached_validation.json').read_text())['pooled']
    np.testing.assert_allclose(pooled[-1],curr['score']-orig['score'],rtol=0,atol=1e-9)
    rng=np.random.default_rng(82713)
    draws=[]
    for _ in range(2500):
        sim=np.zeros_like(total)
        for d in rng.integers(4,size=4):
            blocks=groups[d]
            sim+=blocks[rng.integers(len(blocks),size=len(blocks))].sum(0)
        draws.append(gain(sim))
    report=dict(baseline=clock.OUT.name,rows=int(total[0,0,0]),columns=['weighted_Y2','weighted_Y3','score'],
                pooled=pooled.tolist(),daily=daily,omit_date={d+17:gain(total-s).tolist() for d,s in enumerate(sums)},
                block_ticks=120,bootstrap=2500,interval_95=np.quantile(draws,[.025,.975],axis=0).tolist())
    (joint.OUT/'paired_comparison.json').write_text(json.dumps(report,indent=2))
    print('paired gains',pooled.tolist(),'interval',report['interval_95'],flush=True)


if __name__=='__main__':run()
