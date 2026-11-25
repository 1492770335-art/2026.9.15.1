"""Replay the parent from base forecasts once and verify its two alternative market confidence heads."""
import importlib
import json
import sys
import numpy as np
import torch

import experiment_y3_market_reversal_flow_v1 as parent
import y3_tail_specialist_eval as scoring


def verify(names):
    replay=scoring.trace.replay
    modules=[importlib.import_module(name) for name in names]
    updates=[module.correction() for module in modules]
    rows=[[] for _ in modules]
    reports=[{} for _ in modules]
    old=json.loads((parent.OUT/'reload_verification.json').read_text())

    def on_day(d,raw,p,h):
        n=len(h)
        y=replay.labels(d)
        for j,update in enumerate(updates):
            q=p.copy();q[:,:,2]+=update(raw,p)
            prefix=h.copy();prefix[:,:,2]+=update(raw[:n],h)
            error=float(abs(prefix-q[:n]).max())
            assert error<1e-4 and np.isfinite(q).all()
            np.testing.assert_array_equal(q[:,:,:2],p[:,:,:2])
            row=(p.reshape(-1,3),q.reshape(-1,3),y)
            s=replay.measure([row])
            assert abs(s['baseline_score']-old['daily'][str(d+1)]['score'])<1e-10
            rows[j].append(row)
            reports[j][str(d+1)]=dict(score=s['score'],gain=s['gain'],prefix_error=error)
        print('replayed day',d+1,flush=True)

    base=replay.evaluate(parent.layer(),on_day=on_day)
    np.testing.assert_allclose(base['pooled']['candidate'],old['pooled']['candidate'],rtol=0,atol=1e-10)
    for module,batch,daily in zip(modules,rows,reports):
        result=replay.measure(batch)
        truth=json.loads((module.OUT/'raw_replay.json').read_text())
        assert result['rows']==936250
        np.testing.assert_allclose(result['candidate'],truth['pooled']['candidate'],rtol=0,atol=1e-10)
        for day in daily:assert abs(daily[day]['score']-truth['daily'][day]['score'])<1e-10
        out=module.OUT/'full_chain_verification'
        out.mkdir(exist_ok=True)
        (out/'composed_replay.json').write_text(json.dumps(dict(pooled=result,daily=daily,parent=parent.OUT.name),indent=2))
        print(module.OUT.name,result['score'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    verify(sys.argv[1:])
