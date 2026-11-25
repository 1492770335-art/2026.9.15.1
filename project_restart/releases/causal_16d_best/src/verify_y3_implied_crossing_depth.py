"""Recompute the location contrast by traversing surviving levels in reverse time."""
import importlib
import json
import sys
import numpy as np
import torch
import experiment_y3_implied_crossing_depth_v1 as depth


def run(name):
    model=importlib.import_module(name)
    with np.load(model.OUT/'model.npz') as f:
        st=dict(f)
    assert int(st['pick'])==9
    w=st['w'][:,9]
    cols=np.r_[0:3,111:123]
    assert not np.any(np.delete(w,cols))
    raw=model.trial.clock.book.rawday(15)
    p=model.history.references(15,raw)[0]
    full=model.correction()(raw,p)
    r,m=depth.parent.signals(raw,st)
    _,scores=depth.event.tape(raw,st,depth.event.restore(st))
    implied=scores[:,:,0]*st['scale']
    ticks=np.unique(np.r_[0,1,31,32,95,96,2400,2401,2402,np.arange(16,len(raw),93)])
    diffs=[]
    for t in ticks:
        pieces=[p[t]]
        for h in depth.parent.WINDOWS:
            lo=max(0,t-h+1,2401 if t>=2401 else 0)
            steps=r[lo:t+1]
            before=steps.cumsum(0)-steps
            end=steps.sum(0)
            sign=np.sign(end)
            remain=abs(end)
            features=np.zeros((50,6))
            for j in reversed(range(len(steps))):
                first=sign*before[j]
                width=np.maximum(0,steps[j]*sign)
                stop=first+width
                lower=np.maximum(0,first)
                match=np.clip(implied[lo+j]*sign,0,width)
                head=(np.minimum(remain,first+match)-lower).clip(0)
                tail=(np.minimum(remain,stop)-np.maximum(lower,stop-match)).clip(0)
                # A losing move cannot add any level. The surviving endpoint also cannot exceed this step.
                d=head-tail
                a,b,c,e=m[lo+j].T
                age=(h-1-(t-lo-j))/h
                features+=d[:,None]*np.stack((sign,age*sign,a,b,sign*c,sign*e),1)
                remain=np.minimum(remain,lower)
            pieces.append(features/(np.sqrt(h)*(1+np.sqrt(abs(end)/np.sqrt(h))))[:,None])
        x=np.column_stack(pieces)
        diffs.append(float(abs((x-x.mean(0))@w[cols]-full[t]).max()))
    fn=model.correction()
    stops=(1,32,96,311,2401,2402)
    prefix=max(float(abs(fn(raw[:n],p[:n])-full[:n]).max()) for n in stops)
    summary=dict(day=16,stock_ticks=len(ticks)*50,direct_error=max(diffs),prefixes=len(stops),prefix_error=prefix)
    assert max(max(diffs),prefix)<4e-6,summary
    (model.OUT/'inference_verification.json').write_text(json.dumps(summary,indent=2))
    print(summary,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run(sys.argv[1] if len(sys.argv)>1 else 'experiment_y3_implied_crossing_depth_v1')
