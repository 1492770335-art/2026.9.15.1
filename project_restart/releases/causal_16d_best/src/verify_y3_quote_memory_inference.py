"""Check frozen quote memory without access to training data and verify time ordering in a small trace."""
import json

import numpy as np
import torch

import experiment_y3_same_quote_memory_v1 as model


def run():
    # Load one historical trace before disabling all training-data access.
    raw=model.trial.clock.book.rawday(9)[:841]
    p=model.history.references(9,model.trial.clock.book.rawday(9))[0,:841]
    update=model.correction()
    def forbidden(*args,**kwargs):
        raise AssertionError('frozen inference attempted to access fitting data')
    saved=[(model.trial.clock.book,'rawday'),(model.trial.clock,'labels'),(model.fields,'training_profile'),
           (model.quote,'calibration'),(model.quote.grid,'calibration')]
    originals=[getattr(obj,key) for obj,key in saved]
    try:
        for obj,key in saved:setattr(obj,key,forbidden)
        full=update(raw,p)
        cuts=(1,119,120,121,239,240,241,417,710)
        errors=[float(np.max(np.abs(full[:cut]-update(raw[:cut],p[:cut])))) for cut in cuts]
        assert max(errors)<1e-6
    finally:
        for (obj,key),value in zip(saved,originals):setattr(obj,key,value)
    with np.load(model.OUT/'model.npz') as f:
        mid, width, _, ok=model.quote.infer(raw,float(f['k']),f['coefficients'])
    checked=0
    for level in (mid-width,mid+width):
        prior=model.previous_visits(level,ok)
        for s in range(50):
            last={}
            for t in range(len(raw)):
                if ok[t,s]:
                    price=level[t,s]
                    assert prior[t,s]==last.get(price,-1)
                    last[price]=t
                    checked+=1
                else:
                    assert prior[t,s]==-1
    result=dict(cuts=cuts,errors=errors,checked_valid=checked,training_data_access=False)
    (model.OUT/'frozen_inference_checks.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
