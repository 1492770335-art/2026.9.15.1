"""Repeat the original transition-versus-current-state comparison using cutoff-specific reference forecasts."""
import json
import numpy as np
import experiment_cutoff_reference_panel_v1 as refs
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

state=refs.state
walk=state.walk
OUT=ART/'cutoff_transition_rerun_v1'


def main():
    OUT.mkdir(exist_ok=True)
    history=[];folds=[];daily=[]
    for end in (8,10,12,14):
        w=state.fit(history) if history else None
        groups=[[[],[]] for _ in range(2)]
        for d in range(end,end+2):
            raw=walk.book.rawday(d);y=labels(d).reshape(-1,50,3)
            for j,name in enumerate(('panel','neural')):
                p=np.load(refs.OUT/f'{name}_{d}.npy')
                st,x=state.inputs(raw,p)
                if not j:history.append(state.statistics(p[::4],y[::4],st[:,::4],x[::4]))
                if w is None:continue
                for k in (0,1):
                    q=state.predict(p,st,x,w,k)
                    row=tuple(v.reshape(-1,3) for v in (p,q,y))
                    groups[j][k].append(row)
                    if j==0 and k:
                        s=measure([row]);daily.append(np.subtract(s['candidate'],s['base']))
        if w is not None:
            f=[[measure(v) for v in g] for g in groups];folds.append(f)
            print(end+1,[[v['gain'] for v in g] for g in f],flush=True)
    gain=np.array([[np.subtract(b['candidate'],b['base']) for a,b in f] for f in folds])
    edge=np.array([[np.subtract(b['candidate'],a['candidate']) for a,b in f] for f in folds])
    keep=(gain[:,0].min(0)>0)&(edge[:,0].mean(0)>0)&(gain[:,1].mean(0)>0)&((np.stack(daily)>0).sum(0)>=4)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,retained=keep.tolist(),daily=np.stack(daily).tolist()),indent=2))
    if keep.any():
        np.savez(OUT/'model.npz',weights=state.fit(history),mask=keep)
        verify()


def load():
    with np.load(OUT/'model.npz') as st:w,mask=st['weights'].copy(),st['mask'].copy()
    with np.load(state.OUT/'model.npz') as st:old,prior_mask=st['weights'].copy(),st['mask'].copy()
    parent=walk.specialist.load()
    def predict(raw,p):
        q=parent(raw,p)
        st,x=state.inputs(raw,q)
        base=state.predict(q,st,x,old,True,prior_mask)
        after=state.predict(q,st,x,w,True,mask)
        base[:,:,mask]=after[:,:,mask]
        return base
    return predict


def verify():
    result=walk.replay(load())
    best=json.loads((state.OUT/'reload_verification.json').read_text())['pooled']['score']
    result['gain_over_best']=result['pooled']['score']-best
    path=OUT/'raw_replay.json'
    if path.exists():
        before=json.loads(path.read_text())
        np.testing.assert_allclose(result['pooled']['candidate'],before['pooled']['candidate'],rtol=0,atol=1e-10)
    path.write_text(json.dumps(result,indent=2))
    print(result['pooled'],result['gain_over_best'],flush=True)


if __name__=='__main__':main()
