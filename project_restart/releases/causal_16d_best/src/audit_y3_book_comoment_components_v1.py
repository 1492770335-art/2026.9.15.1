"""Compare the two pre-specified component ablations with the frozen joint readout."""
import json
import numpy as np
import torch
import experiment_y3_future_book_comoments_v1 as model

history,trial,stack=model.history,model.trial,model.stack


def main():
    rows=[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        refs=history.references(d,raw) if d>=8 else [history.study.screen.history.market.prior.path.early.load(d,raw)]
        t=np.arange(d%5,len(raw)-60,5)
        t=t[(t+60<2401)|(t>=2401)]
        z=model.advice(raw,t,model.params(d-d%2))
        rows.append([model.model.stats(p,y,t,z) for p in refs])
    w=model.model.weights(rows,model.GROUPS)[[0,2,5,6]]
    with np.load(model.OUT/'model.npz') as f:
        np.testing.assert_allclose(w[3],f['w'],atol=1e-9,rtol=0)
        st=dict(f)
    trace=stack.verifier.trace
    totals=[]
    for d in range(16,20):
        raw=trace.replay.original.features(trace.replay.original.DATA,d)
        p=history.read(d,raw)[0]
        y=trace.replay.labels(d).reshape(p.shape)
        day=[]
        for t in np.array_split(np.arange(len(raw)),6):
            z=model.advice(raw,t,st)
            updated=p[t,:,2,None]+(p[t]@w[:,:3].T+z@w[:,3:].T).astype('float32')
            a=np.concatenate((np.ones_like(y[t,:,:1]),updated,y[t,:,2:3]),axis=2).reshape(-1,6).astype(float)
            day.append(a.T@a)
        totals.append(sum(day))
    def correlations(m):
        c=stack.covariance(m)
        return c[:-1,-1]/np.sqrt(c.diagonal()[:-1]*c[-1,-1])
    corr=correlations(sum(totals))
    want=json.loads((model.OUT/'reload_verification.json').read_text())['pooled']['candidate'][2]
    assert abs(corr[-1]-want)<1e-9
    report=dict(method='Pre-specified refits using the same historical dates; official four-day row population.',
                models=['incumbent','first moments alone','covariance alone','selected joint readout'],
                Y3=corr.tolist(),gains=(corr-corr[0]).tolist(),daily=[correlations(m).tolist() for m in totals])
    (model.OUT/'component_ablation.json').write_text(json.dumps(report,indent=2))
    print(corr,corr-corr[0],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
