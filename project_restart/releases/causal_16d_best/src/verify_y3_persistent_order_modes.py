"""Recompute the selected Y3 correction directly from frame-by-frame historical orders and prices."""
import json
import numpy as np
import torch
import experiment_y3_persistent_order_modes_v1 as model


def run():
    with np.load(model.OUT/'model.npz') as f:
        state=dict(f)
    assert int(state['pick'])==7
    # Check the structure separately for each of four horizon matrices.
    order_coeff=state['weights'][2].astype(float)
    local_coeff=state['weights'][8].astype(float)
    off_diagonal=np.tile(1-np.eye(50),(10,1))
    assert np.max(abs(local_coeff*off_diagonal[:,None,:]))==0
    combined=order_coeff.reshape(500,-1)
    singular=np.linalg.svd(combined,compute_uv=False)
    assert singular[16]<1e-6*singular[0]
    expected=state['w'][:,7]
    used=np.r_[np.arange(3),np.arange(31,35)]
    assert not np.any(np.delete(expected,used))
    raw=model.trial.clock.book.rawday(15)
    p=model.history.references(15,raw)[0]
    ticks=np.unique(np.r_[0,1,4,5,18,19,20,39,40,np.arange(43,len(raw),41)])
    observed=raw[:,:,0].astype(float)-float(np.float16(.00136))
    ret=np.arcsinh(np.log1p(float(state['k'])*observed)/float(state['k']))
    side=(raw[:,:,54]>state['floors'][1]).astype(float)-(raw[:,:,55]>state['floors'][2])
    series=(np.concatenate((side,ret),1)-state['mu'])/state['sc']
    window=[]
    for t in ticks:
        parts=[]
        for lo,hi in model.process.LAGS:
            portion=series[max(0,t+1-hi):max(0,t+2-lo)]
            parts.append(portion.sum(0)/(hi-lo+1))
        feature=np.concatenate(parts)
        # Compute separate source-to-recipient products before the Y3 projection.
        order=np.einsum('i,ihs->sh',feature,order_coeff)
        own=np.einsum('i,ihs->sh',feature,local_coeff)
        window.append((order+own)*state['sc'][50:,None])
    window=np.stack(window)
    ref=np.concatenate((p[ticks],window),axis=2)
    independent=(ref-ref.mean(1,keepdims=True))@expected[used]
    correct=model.correction()
    full=correct(raw,p)
    np.testing.assert_allclose(independent,full[ticks],atol=4e-6,rtol=0)
    prefix_errors=[]
    for end in (1,2,6,19,20,21,40,41,311,1201,2450,len(raw)-1):
        diff=np.max(abs(correct(raw[:end],p[:end])-full[:end]))
        assert diff<4e-6
        prefix_errors.append(float(diff))
    result=dict(day=16,stock_ticks=len(ticks)*50,max_direct_error=float(abs(independent-full[ticks]).max()),
                prefix_count=len(prefix_errors),max_prefix_error=max(prefix_errors),
                seventeenth_singular_ratio=float(singular[16]/singular[0]))
    (model.OUT/'inference_verification.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
