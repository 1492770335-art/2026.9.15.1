"""Check the optimized inference path against all features, then score it on the independently replayed parent."""
import json
import numpy as np
import torch

import cache_y3_context_reference as reference
import experiment_y3_expanding_path_complement_v1 as model

scoring=reference.scoring


def main():
    with np.load(model.OUT/'model.npz') as f:
        m={k:f[k] for k in f.files if k!='pick'}
        pick=int(f['pick'])
    update=model.correction()
    rows=[]
    daily={}
    diff=0.
    for d in range(16,20):
        raw=scoring.trace.replay.original.features(scoring.trace.replay.original.DATA,d)
        p,h,n=reference.load(d,raw)
        short=model.changes_with_reference(model.encode(raw[:n],m),m,h)[pick]
        fast=update(raw[:n],h)
        diff=max(diff,float(abs(short-fast).max()))
        np.testing.assert_array_equal(short,fast)
        q=p.copy()
        q[:,:,2]+=update(raw,p)
        h[:,:,2]+=fast
        assert np.max(abs(q[:n]-h))<1e-4
        row=(p.reshape(-1,3),q.reshape(-1,3),scoring.trace.replay.labels(d))
        daily[d+1]=scoring.trace.older.measure([row])
        rows.append(row)
    pooled=scoring.trace.older.measure(rows)
    assert pooled['rows']==936250
    baseline=json.loads((model.history.incumbent.OUT/'full_chain_verification/composed_replay.json').read_text())
    np.testing.assert_allclose(pooled['base'],baseline['pooled']['candidate'],rtol=0,atol=1e-10)
    result=dict(pooled=pooled,daily=daily,max_difference=diff)
    (model.OUT/'optimized_readout_check.json').write_text(json.dumps(result,indent=2))
    print(pooled,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
