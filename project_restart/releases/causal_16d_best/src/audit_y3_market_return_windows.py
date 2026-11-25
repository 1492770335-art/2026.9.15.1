"""Measure both common-flow adjustments together and attribute their correlations to future market windows."""
import json
import numpy as np
import torch

import experiment_y3_overshoot_basket_agreement_v1 as model
import y3_unfulfilled_references as history

scoring=history.rolling.cached.scoring


def correlation(g):
    cov=g[1:,1:]-np.outer(g[0,1:],g[0,1:])/g[0,0]
    return cov[0,1:]/np.sqrt(cov[0,0]*cov.diagonal()[1:])


def main():
    base_report=json.loads((history.incumbent.OUT/'reload_verification.json').read_text())
    expected=json.loads((model.OUT/'reload_verification.json').read_text())
    stages=(history.rolling.incumbent,history.previous.parent.previous.incumbent,history.previous.parent.incumbent,
            history.previous.incumbent,history.incumbent)
    correctors=[s.correction() for s in stages]
    market,agree=model.market.correction(),model.correction()
    coef=model.trial.curve.settled.historical_geometry()[1]
    ratio=coef[2,1]/coef[1,1]
    grams={};blocks=[];rows=[]
    for d in range(16,20):
        raw=scoring.trace.replay.original.features(scoring.trace.replay.original.DATA,d)
        p,_,_=history.rolling.cached.load(d,raw)
        for update in correctors: p[:,:,2]+=update(raw,p)
        q=p.copy();q[:,:,2]+=market(raw,q);q[:,:,2]+=agree(raw,q)
        y=scoring.trace.replay.labels(d).reshape(q.shape)
        row=(p.reshape(-1,3),q.reshape(-1,3),y.reshape(-1,3))
        score=scoring.trace.older.measure([row])
        assert abs(score['baseline_score']-base_report['daily'][str(d+1)]['score'])<1e-10
        assert abs(score['score']-expected['daily'][str(d+1)]['score'])<1e-10
        rows.append(row)
        blocks.append(scoring.audit.blockstats(p,y,[q[:,:,2].astype(float)-p[:,:,2]]))
        a,b,c=(x.astype(float).mean(1) for x in (y,p,q))
        def add(name,target,old,new):
            z=np.column_stack((np.ones(len(target)),target,old,new))
            grams[name]=grams.get(name,0)+z.T@z
        add('market_y3',a[:,2],b[:,2],c[:,2])
        before,after=b[:,2]-ratio*b[:,1],c[:,2]-ratio*c[:,1]
        add('market_last40',a[:,2]-ratio*a[:,1],before,after)
        t=np.arange(len(y)-60)
        for offset in (20,40):
            add('market_window_'+str(offset+1)+'_'+str(offset+20),a[t+offset,1],before[t],after[t])
    summary=scoring.trace.older.measure(rows)
    report=dict(pooled=summary, market_correlations={k:correlation(g).tolist() for k,g in grams.items()},
                score_delta_percentiles=(.5*np.asarray(scoring.audit.intervals(blocks))[:,0]).tolist())
    (model.OUT/'combined_market_windows.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
