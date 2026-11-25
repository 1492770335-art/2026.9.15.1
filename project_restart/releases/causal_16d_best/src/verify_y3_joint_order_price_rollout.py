"""Compare the frozen four-interval return forecast with explicit forward recursion."""
import json
import numpy as np
import torch
import experiment_y3_joint_order_price_rollout_v1 as model


def forward(state, t, ar):
    history = np.zeros((40, 100))
    for lo, hi in model.AGES:
        available = state[max(0,t-hi):max(0,t-lo+1)]
        history[39-hi:40-lo] = available.sum(0)/(hi-lo+1)
    past = list(history)
    for _ in range(60):
        following = np.zeros(100)
        for matrix, (lo,hi) in zip(ar,model.process.LAGS):
            following += np.mean(past[len(past)-hi:len(past)-lo+1], axis=0) @ matrix
        past.append(following)
    future = np.array(past[40:])[:,50:]
    return np.stack([future[lo-1:hi].sum(0) for lo,hi in model.process.BANDS],1)


def run():
    with np.load(model.OUT/'model.npz') as f:
        state=dict(f)
    assert state['pick']==7
    fitted=model.process.fitted(16)
    ar=fitted['ar'].astype(float).reshape(5,100,100)
    raw=model.trial.clock.book.rawday(15)
    p=model.history.references(15,raw)[0]
    t=np.unique(np.r_[0,1,3,6,20,38,39,40,np.arange(41,len(raw),43)])
    x=(model.process.state(raw,state)-state['mu'])/state['sc']
    explicit=np.stack([forward(x,i,ar) for i in t])*state['sc'][None,50:,None]
    efficient=model.estimates(raw,t,state)[:,1].transpose(0,2,1)
    np.testing.assert_allclose(explicit,efficient,atol=8e-5,rtol=2e-5)
    w=state['w'][:,7]
    # The selected projection uses the reference and the four full-process windows.
    selected=np.r_[np.arange(3),np.arange(17,21)]
    assert np.all(np.delete(w,selected)==0)
    independent=model.trial.clock.center(np.concatenate((p[t],explicit),2)).astype('float32')@w[selected]
    update=model.correction()
    deployed=update(raw,p)
    np.testing.assert_allclose(independent,deployed[t],atol=4e-6,rtol=0)
    prefix_error=0.
    stops=(1,2,8,21,39,40,41,311,941,2431,len(raw)-1)
    for stop in stops:
        subset=update(raw[:stop],p[:stop])
        diff=float(np.max(abs(subset-deployed[:stop])))
        assert diff<4e-6
        prefix_error=max(prefix_error,diff)
    report=dict(sample_days=[16],ticks=len(t),stock_ticks=len(t)*50,prefixes=stops,
                max_window_error=float(abs(explicit-efficient).max()),max_correction_error=float(abs(independent-deployed[t]).max()),
                max_prefix_error=prefix_error)
    (model.OUT/'forward_recursion_verification.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
