"""Independently replay source-price revisions, routing and rolling readout with frozen weights."""
import json
from unittest.mock import patch
import numpy as np
import torch

import experiment_y3_peer_terminal_price_news_v1 as model


def run():
    day=model.trial.clock.book.rawday(11)
    p=model.history.references(11,day)[1,:743]
    raw=day[:len(p)]
    with np.load(model.OUT/'model.npz') as f:
        st=dict(f)
    assert int(st['pick'])==9
    fn=model.correction()
    with patch.object(model,'params',side_effect=AssertionError('fit attempted')), \
         patch.object(model.trial.clock.book,'rawday',side_effect=AssertionError('train data accessed')), \
         patch.object(model.trial.clock,'labels',side_effect=AssertionError('label access')):
        q=fn(raw,p)
        cuts=(1,2,5,20,21,59,60,61,241,449,512)
        errors=[float(np.max(abs(fn(raw[:c],p[:c])-q[:c]))) for c in cuts]
    mid,half,_,_,ok,_,_,_=model.routing.queues.tape.prices(raw,st)
    active=(np.diff(mid,axis=0)!=0)|(np.diff(half,axis=0)!=0)
    coeff=st['coeff']
    k=float(st['k'])
    w=st['w'][:,9]
    indices=np.r_[0:3,6:15]
    np.testing.assert_array_equal(np.delete(w,indices),0)
    graph=model.routing.balanced(st['graphs'][1])
    log_levels=np.log1p((k*(p-coeff[:,0])/coeff[:,1]).clip(-.95,None))*coeff[2,1]/k
    spot=np.log1p(k*(raw[:,:,0].astype(float)-.00136))*coeff[2,1]/k
    frames=[]
    result=[]
    for i in range(len(p)):
        messages=np.zeros((3,50))
        if i:
            for s in np.flatnonzero(active[i-1]&ok[i]&ok[i-1]):
                now=np.r_[0,log_levels[i,s]]
                past=np.r_[0,log_levels[i-1,s]]
                for j,h in enumerate((5,20,40)):
                    v=spot[i,s]+np.interp(h,(0,5,20,60),now)-np.interp(h+1,(0,5,20,60),past)
                    messages[j,s]=np.arcsinh(v)
        frames.append(messages@graph)
        feat=np.column_stack((p[i],*[np.sum(frames[-w:],0)[j]/np.sqrt(w) for j in range(3) for w in (5,20,60)]))
        result.append((feat-feat.mean(0)).astype('float32')@w[indices])
    loop=float(np.max(abs(np.stack(result)-q)))
    assert max([loop,*errors])<1e-6
    doc=dict(cuts=cuts,prefix_errors=errors,independently_replayed_rows=len(q)*50,step_max_error=loop)
    (model.OUT/'frozen_inference_checks.json').write_text(json.dumps(doc,indent=2))
    print(doc,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
