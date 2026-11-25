"""Ablate fixed midpoint risk and forecast reweighting, keeping all fitted amplitudes frozen."""
import json
import numpy as np
import torch

import joint_interval_covariance_scale_references as latest

head = latest.incumbent
bridge = latest.previous.incumbent
baseline = latest.previous.previous


def main():
    trace = latest.stack.verifier.trace
    with np.load(bridge.OUT/'model.npz') as f:
        state = dict(f)
    w = state['w']
    assert np.array_equal(np.flatnonzero(w[3:]),[4,5])
    scale = head.adjustment()
    arms = ('forecast weights','middle covariance','last covariance','both covariance terms','all terms')
    groups = ((0,),(1,),(2,),(1,2),(0,1,2))
    rows = [[] for _ in arms]
    for d in range(16,20):
        raw = trace.replay.original.features(trace.replay.original.DATA,d)
        p,_,_ = baseline.read(d,raw)
        q,_,_ = latest.read(d,raw)
        parts = []
        for t in np.array_split(np.arange(len(raw)),max(1,(len(raw)+499)//500)):
            z = bridge.advice(raw,t,state)
            parts.append(np.stack((p[t]@w[:3],z[:,:,4]*w[7],z[:,:,5]*w[8]),axis=2))
        v = np.concatenate(parts)
        y = trace.replay.labels(d)
        for j,g in enumerate(groups):
            inter = p.copy()
            inter[:,:,2] += v[:,:,list(g)].sum(2)
            result = scale(p,inter)
            if j==len(groups)-1:
                np.testing.assert_allclose(result,q,rtol=0,atol=2.e-6)
            rows[j].append((p.reshape(-1,3),result.reshape(-1,3),y))
    scores = [trace.older.measure(r) for r in rows]
    expected = json.loads((head.OUT/'reload_verification.json').read_text())['pooled']['score']
    assert abs(scores[-1]['score']-expected)<1.e-9
    report = dict(reference=baseline.incumbent.OUT.name,arms={k:v for k,v in zip(arms,scores)},
                  scope='Post-training attribution with fixed amplitudes on four complete validation dates.')
    (head.OUT/'frozen_component_ablation.json').write_text(json.dumps(report,indent=2))
    print([(k,s['gain']) for k,s in zip(arms,scores)],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
