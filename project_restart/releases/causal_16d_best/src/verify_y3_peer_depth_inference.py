"""Independently replay the selected Y3 peer-pressure correction one observation at a time."""
import json
from unittest.mock import patch

import numpy as np
import torch

import experiment_y3_peer_depth_event_causes_v1 as model


def check():
    raw=model.trial.clock.book.rawday(9)[:811]
    p=model.history.references(9,model.trial.clock.book.rawday(9))[0,:len(raw)]
    with np.load(model.OUT/'model.npz') as f:
        st=dict(f)
    pick=int(st['pick'])
    assert pick==10
    g=model.source.balanced(st['graphs'][1])
    np.testing.assert_allclose(g.diagonal(),0,atol=1e-10)
    np.testing.assert_allclose(g.sum(0),0,atol=2e-7)
    mid,half,_,_,valid,_,buy,sell=model.source.queues.tape.prices(raw,st)
    bid,ask=model.source.queues.size(raw,st)
    fn=model.correction()
    with patch.object(model.trial.clock.book,'rawday',side_effect=AssertionError('training data accessed')), \
         patch.object(model.trial.clock,'labels',side_effect=AssertionError('labels accessed')), \
         patch.object(model.source,'params',side_effect=AssertionError('attempted to refit')):
        reference=fn(raw,p)
        cuts=(1,2,5,19,20,21,59,60,61,311,543)
        errors=[float(abs(fn(raw[:n],p[:n])-reference[:n]).max()) for n in cuts]
    weights=st['w'][:,pick]
    idx=np.r_[0:3,18:21,12:15,15:18]
    np.testing.assert_array_equal(np.delete(weights,idx),0)
    messages=[]
    preds=[]
    for t in range(len(raw)):
        x=np.zeros((50,3))
        if t:
            for s in range(50):
                if not (valid[t,s] and valid[t-1,s]):continue
                for direction,level,q,v in ((1,mid[:,s]-half[:,s],bid[:,s],sell[t,s]),
                                             (-1,mid[:,s]+half[:,s],ask[:,s],buy[t,s])):
                    move=(level[t]-level[t-1])*direction
                    if move==0:x[s,0]+=direction*(q[t]-q[t-1]+v)
                    elif move>0:x[s,1]+=direction*q[t]
                    else:x[s,2]-=direction*q[t-1]
        messages.append(np.arcsinh(x/st['volume_scale'][:,None]))
        block=[]
        for j in range(3):
            block.extend([np.sum(messages[-w:],axis=0)[:,j]@g/np.sqrt(w) for w in (5,20,60)])
        features=np.column_stack((p[t],*block))
        preds.append((features-features.mean(0)).astype('float32')@weights[idx])
    loop=float(abs(np.stack(preds)-reference).max())
    assert max([loop,*errors])<1e-6
    report=dict(prefixes=cuts,prefix_max_abs_errors=errors,loop_frames=len(raw),stocks=50,loop_max_error=loop)
    (model.OUT/'frozen_inference_checks.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    check()
