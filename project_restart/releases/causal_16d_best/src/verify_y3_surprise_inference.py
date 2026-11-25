"""Check production readout against all counterfactual features and the four daily validation scores."""
import json
import numpy as np
import torch

import experiment_y3_expected_price_paths_v1 as model
import y3_expanding_path_references as rolling


def main():
    with np.load(model.OUT/'model.npz') as f:
        m=dict(f)
    report=json.loads((model.OUT/'reload_verification.json').read_text())
    scoring=rolling.cached.scoring.trace
    stages=(rolling.incumbent,model.history.previous.incumbent,model.history.incumbent)
    parent=[stage.correction() for stage in stages]
    update=model.correction()
    rows=[]
    for d in range(16,20):
        raw=scoring.replay.original.features(scoring.replay.original.DATA,d)
        p,_,_=rolling.cached.load(d,raw)
        for stage in parent:
            p[:,:,2]+=stage(raw,p)
        full=model.changes_with_reference(model.encode(raw,m),m,p)[int(m['pick'])]
        reduced=update(raw,p)
        np.testing.assert_array_equal(reduced,full)
        q=p.copy()
        q[:,:,2]+=reduced
        row=(p.reshape(-1,3),q.reshape(-1,3),scoring.replay.labels(d))
        assert abs(scoring.older.measure([row])['score']-report['daily'][str(d+1)]['score'])<1e-10
        rows.append(row)
    s=scoring.older.measure(rows)
    assert s['rows']==936250
    np.testing.assert_allclose(s['candidate'],report['pooled']['candidate'],rtol=0,atol=1e-10)
    (model.OUT/'optimized_check.json').write_text(json.dumps(dict(pooled=s,max_difference=0.),indent=2))
    print(s,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
