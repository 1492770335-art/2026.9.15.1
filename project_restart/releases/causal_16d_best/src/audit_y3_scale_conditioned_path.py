"""Ablate direction and stock-specific magnitude using the selected readout weights unchanged."""
import json
import numpy as np
import torch

import cache_y3_context_reference as cached
import experiment_y3_scale_conditioned_path_v1 as model

NAMES=('frozen path','direction only','own magnitude','other stock magnitude','mean magnitude')


def main():
    report=json.loads((model.OUT/'reload_verification.json').read_text())
    with np.load(model.OUT/'model.npz') as f:
        weights=dict(f)
    pick=int(weights['pick'])
    assert pick==2
    w=weights['w'][:,pick]
    assert not np.any(np.delete(w,[0,1,2,7,8]))
    update_parent=model.base.correction()
    official=cached.scoring.trace
    stats={name:[] for name in NAMES}
    daily={}
    for d in range(16,20):
        raw=official.replay.original.features(official.replay.original.DATA,d)
        p,_,_=cached.load(d,raw)
        p[:,:,2]+=update_parent(raw,p)
        y=official.replay.labels(d).reshape(p.shape)
        features=model.encode(raw,weights)
        deltas=[]
        for j in (0,1,2,4,6):
            copy=features.copy()
            copy[:,:,4:6]=features[:,:,2*j:2*j+2]
            deltas.append(model.changes_with_reference(copy,weights,p)[pick])
        daily[d+1]={}
        for name,delta in zip(NAMES,deltas):
            q=p.copy()
            q[:,:,2]+=delta
            row=(p.reshape(-1,3),q.reshape(-1,3),y.reshape(-1,3))
            stats[name].append(row)
            daily[d+1][name]=official.older.measure([row])['score']
        assert abs(daily[d+1]['own magnitude']-report['daily'][str(d+1)]['score'])<1e-10
    scores={name:official.older.measure(rows) for name,rows in stats.items()}
    assert scores['own magnitude']['rows']==936250
    np.testing.assert_allclose(scores['own magnitude']['candidate'],report['pooled']['candidate'],atol=1e-10,rtol=0)
    result=dict(weights='same selected Y3 readout in every intervention',pooled=scores,daily=daily)
    (model.OUT/'magnitude_ablation.json').write_text(json.dumps(result,indent=2))
    print({n:a['score'] for n,a in scores.items()},flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
