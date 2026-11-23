"""Aggregate expert revisions whose direction replicates on both earlier chronological training windows."""
import json
import numpy as np
import torch
import experiment_delayed_tail_band_skill_v1 as early
import experiment_eight_date_expert_contrasts_v1 as experts
import experiment_replicated_revision_slope_v1 as slope
import experiment_phase_future_book_v1 as phase
import experiment_symmetric_pressure_network_v1 as sym
import experiment_pairwise_spread_decoder_v1 as spread
import experiment_shared_bar_gru_v1 as bar
import replay_verified_forecast_parent_v1 as replay
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

PARENT=slope.OUT
OUT=ART/'replicated_expert_consensus_v1'


def fit():
    daily=[]
    for d in range(8,16):
        p=early.base(d)
        daily.append([slope.day_weights(p,np.load(ART/name/f'side_{d}.npy'),labels(d)) for name in experts.SOURCES])
    d=np.stack(daily)
    signs=np.sign(d)
    direction=signs[-1]
    ok=(signs[4:]==direction).all(0)&((signs[:4]==direction).sum(0)>=3)
    # The pairwise spread model has no market head.
    ok[2,0]=False
    size=np.minimum(np.quantile(d[:4]*direction,.25,axis=0),np.min(d[4:]*direction,axis=0)).clip(0,.4)
    return .7*size*direction*ok/ok.sum(0).clip(1),d


def advance(p,sources,coeff):
    q=p.astype(float)
    for n,w in zip(sources,coeff):
        m,r=slope.parts(p,n)
        q+=m*w[0]+r*w[1]
    return q.astype('float32')


def sources():
    functions=[]
    for i,(name,cls,key) in enumerate(zip(experts.SOURCES,(phase.BookNet,sym.PressureNet,spread.PairNet,bar.ScaleNet),
                                         ('current.0.weight','gate.0.weight','state.0.weight',None))):
        st=torch.load(ART/name/'net_16.pth',map_location='cpu',weights_only=True)
        net=(cls(st[key].shape[1]) if key else cls()).cuda().eval()
        net.load_state_dict(st)
        if i==1:
            fn=lambda raw,model=net:phase.infer(model,raw,make_features=sym.features)
        elif i==3:
            fn=lambda raw,model=net:bar.predict(model,raw)
        else:
            fn=lambda raw,model=net:phase.infer(model,raw)
        functions.append(fn)
    return functions


def main():
    OUT.mkdir(exist_ok=True)
    w,daily=fit()
    np.save(OUT/'coefficients.npy',w)
    rows=[]
    for d in range(16,20):
        p=np.load(PARENT/f'pred_{d}.npy')
        q=advance(p,[np.load(ART/name/f'side_{d}.npy') for name in experts.SOURCES],w)
        np.save(OUT/f'pred_{d}.npy',q)
        rows.append((p.reshape(-1,3),q.reshape(-1,3),labels(d)))
    result=measure(rows)
    (OUT/'results.json').write_text(json.dumps({'20':result,'weights':w.tolist(),'fit_date_ids':list(range(8,16))},indent=2))
    print(result,flush=True)
    if result['gain']>0:
        models=sources()
        replay.main(PARENT,OUT,lambda raw,p:advance(p,[fn(raw) for fn in models],w))


if __name__=='__main__':
    main()
