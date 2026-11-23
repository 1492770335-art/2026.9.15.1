"""Apply the train-selected individual-stock Y3 correction to the propagated-kernel forecaster."""
import json
import numpy as np
import experiment_individual_response_dynamics_v1 as ind
import experiment_propagated_return_kernel_v1 as parent
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'individual_kernel_forecast_v1'


def frozen():
    with np.load(ind.OUT/'model.npz') as st:
        g={k:st[k].copy() for k in ('center','scale','omega')}
        w=st['w'].copy();params=(st['stable'].copy(),st['drift'].copy());coef=st['coef'].copy()
    def refine(raw,p):
        signal=ind.signals(raw,p,g,w,params)
        q=p.copy();q[:,:,2]+=sum(a@b for a,b in zip(signal,coef))
        return q
    return refine


def main():
    OUT.mkdir(exist_ok=True);f=frozen();rows=[];daily={}
    for d in range(16,20):
        p=np.load(parent.OUT/f'pred_{d}.npy');raw=ind.pair.rawday(d);q=f(raw,p)
        if d==16:np.testing.assert_allclose(q[:350],f(raw[:350],p[:350]),atol=2.e-6)
        np.save(OUT/f'pred_{d}.npy',q)
        row=tuple(v.reshape(-1,3) for v in (p,q,labels(d)));rows.append(row);daily[d+1]=measure([row])
    score=measure(rows)
    (OUT/'results.json').write_text(json.dumps({'20':score,'daily':daily},indent=2));print(score,flush=True)


if __name__=='__main__':
    import sys
    if '--replay' in sys.argv:
        import replay_verified_forecast_parent_v1 as evaluator
        evaluator.main(parent.OUT,OUT,frozen())
    else:main()
