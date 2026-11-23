"""Complete missing longer-horizon components of frozen stage updates using training-only conditional covariance."""
import json
import numpy as np
import experiment_delayed_tail_band_skill_v1 as ref
import replay_verified_forecast_parent_v1 as replay
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'joint_horizon_completion_v1'
PARENT=ART/'inventory_band_fusion_v1'
STAGES=(
 ('individual_kernel_forecast_v1','quote_lattice_replenishment_v1','quote_lattice_replenishment_v1'),
 ('side_network_execution_fusion_v1','execution_location_markouts_v1','execution_location_markouts_v1'),
 ('execution_location_markouts_v1','persistent_execution_state_v1','persistent_execution_state_v1'),
 ('persistent_execution_state_v1','reconstructed_depth_flow_v1','reconstructed_depth_flow_v1'),
 ('reconstructed_depth_flow_v1','damped_depth_fusion_v1','damped_market_state_v1'),
 ('damped_depth_fusion_v1','price_band_depth_curve_v1','price_band_depth_curve_v1'),
)
ROOTS=tuple(dict.fromkeys(v for a,b,_ in STAGES for v in (a,b)))


def residual_covariances():
    cov=[]
    for d in range(8,16):
        p=ref.base(d);y=labels(d).reshape(p.shape)
        pm=p.mean(1,keepdims=True);ym=y.mean(1,keepdims=True);parts=[]
        for pred,real in ((pm,ym),(p-pm,y-ym)):
            a=pred.reshape(-1,3).astype(float);b=real.reshape(-1,3).astype(float)
            a-=a.mean(0);b-=b.mean(0)
            scale=((a*b).sum(0)/(a*a).sum(0)).clip(.4,2.5)
            e=b/scale-a;parts.append(e.T@e/len(e))
        cov.append(parts)
    return np.array(cov)


def coefficient(stages,cov):
    weights=np.zeros((len(stages),2,3,3))
    for k,(_,_,src) in enumerate(stages):
        mask=np.array(json.loads((ART/src/'results.json').read_text())['eligible'])
        for j in range(2):
            for h in (1,2):
                if mask[j,h]:continue
                seen=np.flatnonzero(mask[j,:h])
                if len(seen)==0:continue
                sample=[]
                for g in cov[:,j]:
                    sq=g[np.ix_(seen,seen)];sample.append(np.linalg.solve(sq+np.diag(sq.diagonal())*.025,g[seen,h]))
                weights[k,j,seen,h]=np.median(sample,axis=0)
    return weights


def apply(p,models,w,stages=STAGES):
    out=p.copy()
    for (a,b,_),ws in zip(stages,w):
        change=models[b]-models[a];shared=change.mean(1,keepdims=True)
        out+=((shared@ws[0])+(change-shared)@ws[1]).astype('float32')
    return out


def main():
    OUT.mkdir(exist_ok=True);w=coefficient(STAGES,residual_covariances());np.save(OUT/'completion.npy',w)
    rows=[]
    for d in range(16,20):
        p=np.load(PARENT/f'pred_{d}.npy');src={v:np.load(ART/v/f'pred_{d}.npy') for v in ROOTS}
        q=apply(p,src,w);np.save(OUT/f'pred_{d}.npy',q)
        rows.append((p.reshape(-1,3),q.reshape(-1,3),labels(d)))
    result=measure(rows);print(result,flush=True)
    (OUT/'results.json').write_text(json.dumps({'20':result,'stages':STAGES,'coefficients':w.tolist()},indent=2))
    if result['gain']>0:replay.main(PARENT,OUT,lambda raw,p,*src:apply(p,dict(zip(ROOTS,src)),w),tuple(ART/v for v in ROOTS))

if __name__=='__main__':main()
