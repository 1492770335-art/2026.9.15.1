"""Recompute the selected crossing features by walking back from the terminal level."""
import json
import numpy as np
import torch

import experiment_y3_crossed_level_book_v1 as model


def run():
    with np.load(model.OUT/'model.npz') as f:
        st = dict(f)
    assert int(st['pick'])==5
    cols = np.r_[0:3,63:75]
    w = st['w'][:,5]
    assert np.all(np.delete(w,cols)==0)
    raw = model.trial.clock.book.rawday(15)
    p = model.history.references(15,raw)[0]
    full = model.correction()(raw,p)
    r,marks = model.signals(raw,st)
    ticks = np.unique(np.r_[0,1,30,31,32,94,95,96,2400,2401,2402,np.arange(18,len(raw),91)])
    errors = []
    for t in ticks:
        features = [p[t]]
        for h in model.WINDOWS:
            lo = max(0,t-h+1,2401 if t>=2401 else 0)
            moves = r[lo:t+1]
            before = moves.cumsum(0)-moves
            end = moves.sum(0)
            sign = np.sign(end)
            remaining = abs(end)
            z = np.zeros((50,6))
            for k in reversed(range(len(moves))):
                # Level width reclaimed while tracing back to the start of the move.
                lower = np.minimum(remaining,np.maximum(0,sign*before[k]))
                length = remaining-lower
                remaining = lower
                a,b,c,d = marks[lo+k].T
                size = length*(sign*a>0)
                age = (h-1-(t-lo-k))/h
                z += size[:,None]*np.stack((sign,age*sign,a,b,sign*c,sign*d),1)
            assert max(remaining)<1.e-8
            z /= (np.sqrt(h)*(1+np.sqrt(abs(end)/np.sqrt(h))))[:,None]
            features.append(z)
        x=np.column_stack(features)
        errors.append(np.max(abs((x-x.mean(0))@w[cols]-full[t])))
    prefixes = (1,32,95,96,311,2401,2402)
    fixed = model.correction()
    prefix_error = max(float(np.max(abs(fixed(raw[:n],p[:n])-full[:n]))) for n in prefixes)
    direct = float(max(errors))
    assert max(direct,prefix_error)<4e-6
    data = dict(day=16,stock_ticks=len(ticks)*50,direct_error=direct,prefixes=len(prefixes),prefix_error=prefix_error)
    (model.OUT/'inference_verification.json').write_text(json.dumps(data,indent=2))
    print(data,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
