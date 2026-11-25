"""Verify the common recipient space and reproduce its frozen Y3 correction one origin at a time."""
import json

import numpy as np
import torch

import experiment_y3_receiver_price_portfolios_v1 as model


def run():
    with np.load(model.OUT/'model.npz') as archive:
        state=dict(archive)
    assert int(state['pick'])==8
    shared=state['weights'][4].astype(float)
    local=state['weights'][9].astype(float)
    singular=np.linalg.svd(shared.reshape(-1,50),compute_uv=False)
    assert singular[8]/singular[0]<1e-6
    # The same recipient basis applies to both terminal windows. The local branch must be diagonal.
    diag=np.tile(np.eye(50),(10,1))
    assert np.max(abs(local*(1-diag[:,None,:])))==0
    w=state['w'][:,8]
    cols=np.array([0,1,2,19,20])
    assert np.all(np.delete(w,cols)==0)
    raw=model.trial.clock.book.rawday(15)
    p=model.history.references(15,raw)[0]
    side=(raw[:,:,54]>state['floors'][1]).astype(float)-(raw[:,:,55]>state['floors'][2])
    k=float(state['k'])
    ret=np.arcsinh(np.log1p(k*(raw[:,:,0].astype(float)-float(np.float16(.00136))))/k)
    series=(np.column_stack((side,ret))-state['mu'])/state['sc']
    ticks=np.unique(np.r_[0,1,4,5,18,19,20,39,40,np.arange(57,len(raw),39)])
    predictions=[]
    for t in ticks:
        parts=[]
        for lo,hi in model.process.LAGS:
            past=series[max(0,t-hi+1):max(0,t-lo+2)]
            parts.append(past.sum(axis=0)/(hi-lo+1))
        x=np.concatenate(parts)
        q=np.stack([x@local[:,h,:]+x@shared[:,h,:] for h in (2,3)],1)*state['sc'][50:,None]
        inputs=np.column_stack((p[t],q))
        predictions.append((inputs-inputs.mean(axis=0))@w[cols])
    update=model.correction()
    full=update(raw,p)
    explicit=float(abs(np.stack(predictions)-full[ticks]).max())
    assert explicit<4e-6
    prefixes=(1,2,5,6,19,20,21,39,40,41,331,2401,len(raw)-1)
    prefix=max(float(np.max(abs(update(raw[:n],p[:n])-full[:n]))) for n in prefixes)
    assert prefix<4e-6
    result=dict(day=16,stock_ticks=len(ticks)*50,max_direct_error=explicit,prefix_count=len(prefixes),
                max_prefix_error=prefix,ninth_singular_ratio=float(singular[8]/singular[0]))
    (model.OUT/'inference_verification.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
