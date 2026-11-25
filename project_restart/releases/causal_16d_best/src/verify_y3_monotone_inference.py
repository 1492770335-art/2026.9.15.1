"""Compare full and reduced cross-sectional readouts on four full-day and prefix predictions."""
import json
import numpy as np
import torch
import experiment_y3_monotone_horizon_projection_v1 as model

def main():
    screen=model.screen
    with np.load(model.OUT/'model.npz') as f:w,pick=f['w'],int(f['pick'])
    optimized=model.correction()
    expected=json.loads((model.OUT/'reload_verification.json').read_text())
    rows=[]
    for d in range(16,20):
        trace=screen.cache.scoring.trace
        raw=trace.replay.original.features(trace.replay.original.DATA,d)
        p,h,n=screen.cache.load(d,raw)
        for x,base in ((raw,p),(raw[:n],h)):
            direct=screen.changes(model.features(x,base),base,w)[pick]
            np.testing.assert_array_equal(direct,optimized(x,base))
        q=p.copy();q[:,:,2]+=optimized(raw,p)
        row=(p.reshape(-1,3),q.reshape(-1,3),trace.replay.labels(d))
        assert abs(trace.older.measure([row])['score']-expected['daily'][str(d+1)]['score'])<1e-10
        rows.append(row)
    pooled=trace.older.measure(rows)
    assert pooled['rows']==936250
    np.testing.assert_allclose(pooled['candidate'],expected['pooled']['candidate'],rtol=0,atol=1e-10)
    (model.OUT/'reduced_inference.json').write_text(json.dumps(dict(pooled=pooled,max_difference=0.),indent=2))
    print(pooled,flush=True)

if __name__=='__main__':
    torch.set_num_threads(2)
    main()
