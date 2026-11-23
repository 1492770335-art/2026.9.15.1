"""Calibrate accumulated model revisions against a causally replayable sequence backbone."""
import json
import numpy as np
import experiment_eight_date_expert_contrasts_v1 as readout
import experiment_repricing_adjusted_book_flow_v1 as flow
import package_opening_expert_v1 as opening
import replay_verified_forecast_parent_v1 as replay
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,corr,measure

PARENT=ART/'first_passage_return_v1'
OUT=ART/'backbone_revision_size_v1'


def features(p,opening_forecast):
    pm=p.mean(1,keepdims=True);om=opening_forecast.mean(1,keepdims=True)
    rel=p-pm;ro=opening_forecast-om
    m=om-pm;r=ro-rel
    market=[m[:,0]];local=[r]
    discrepancy=np.sqrt((r*r).mean(1,keepdims=True)/(.008+(rel*rel).mean(1,keepdims=True)))
    agree=np.tanh(ro*rel/(.03+.35*(ro*ro+rel*rel)))
    for k in (0,1,2):
        scale=np.tanh(discrepancy[:,:,k:k+1]-.5)
        market.append((m*scale)[:,0])
        local.append(r*scale)
    local.append(r*agree)
    z=np.concatenate(local,axis=2)
    return np.column_stack(market),z-z.mean(1,keepdims=True)


def main():
    OUT.mkdir(exist_ok=True)
    history=[];report={}
    for end,stop in ((12,14),(14,16),(16,20)):
        if end>=14:
            w=flow.fit(history)
        if end==16:
            mask=np.min(daily,axis=0)>0
            report['eligible']=mask.tolist()
            (OUT/'results.json').write_text(json.dumps(report,indent=2))
            if not mask.any():
                return
            w=[v*gate for v,gate in zip(w,mask)]
            np.savez(OUT/'weights.npz',market=w[0],relative=w[1])
        rows=[];daily=[]
        for d in range(end,stop):
            root=PARENT if end==16 else ART/'intraday_relative_calibration_v1'
            p=np.load(root/f'pred_{d}.npy')
            o=np.load(ART/'opening_oof_cache_v1'/f'day_{d:02d}.npy')
            m,r=features(p,o)
            if end>=14:
                terms=((m@w[0])[:,None],r@w[1])
                q=(p+sum(terms)).astype('float32')
                if end==16:
                    np.save(OUT/f'pred_{d}.npy',q)
            y=labels(d).reshape(p.shape)
            if end<16:
                history.append(readout.moments(p,y,(m,r)))
            if end>=14:
                row=tuple(a.reshape(-1,3) for a in (p,q,y))
                rows.append(row)
                if end==14:
                    base=corr(row[0],row[2])
                    daily.append([corr((p+v).reshape(-1,3),row[2])-base for v in terms])
        if end>=14:
            report[str(stop)]=measure(rows)
            print(stop,report[str(stop)],flush=True)
            (OUT/'results.json').write_text(json.dumps(report,indent=2))
    if report['20']['gain']>0:
        folder=ART/'opening_expert_v1/commit_opening_expert'
        ev=opening.Evaluator('',str(folder),'cuda:0')
        mix=np.array(json.loads((folder/'conf/Model_0.json').read_text())['opening_mix'])
        def advance(raw,p):
            (_,o),_=opening.replay_features(ev,raw,mix,2)
            m,r=features(p,o.reshape(p.shape))
            return (p+(m@w[0])[:,None]+r@w[1]).astype('float32')
        replay.main(PARENT,OUT,advance)


if __name__=='__main__':
    main()
