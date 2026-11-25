"""Separate gains from refreshing inter-stock modes and from refreshing the final residual calibration."""
import json
import numpy as np
import torch

import experiment_y3_persistent_order_modes_v1 as model

study,trial=model.study,model.trial


def design(raw,p,t,st):
    signals=model.forecast(raw,t,st)
    return trial.clock.center(np.concatenate((p[t],signals[8]+signals[2]),2)).astype('float32')


def fit(rows):
    return model.book.prev.gram(rows,[(np.arange(3,7),3,.35)])['w'][:,0]


def run():
    rolling_rows=[]
    fixed_rows=[]
    folds=[]
    regions=[]
    scale_factors=[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        refs=(model.history.references(d,raw) if d>=8 else [study.screen.history.market.prior.path.early.load(d,raw)])
        train=np.arange(90+d%21,len(raw)-60,21)
        evaluate=np.arange(d%4,len(raw)-60,4)
        ticks=np.unique(np.r_[train,evaluate]) if d>=8 else train
        train_index=np.searchsorted(ticks,train)
        st=model.params(max(4,d-d%2))
        if d>=8 and d%2==0:
            w=fit(rolling_rows)
            wf=fit(fixed_rows)
            if d==8:
                initial=st
                initial_readout=w.copy()
            # Use completed dates to separate calibration size from its direction in forecast space.
            covariance=np.mean([a.T@a/len(a) for x,_ in rolling_rows for a in [x.reshape(-1,7).astype(float)]],axis=0)
            norm=lambda v:np.sqrt(v@covariance@v)
            scale=norm(initial_readout)/max(1e-10,norm(w))
            scale_factors.append(float(scale))
            sums=[[],[]]
        fresh_records=[]
        fixed_records=[]
        for j,p in enumerate(refs):
            x=design(raw,p,ticks,st)
            fixed=(x if d<10 else design(raw,p,ticks,initial))
            err=study.error(p,y,train)
            fresh_records.append((x[train_index],err))
            fixed_records.append((fixed[train_index],err))
            if d<8:
                continue
            ix=np.searchsorted(ticks,evaluate)
            old,new=fixed[ix],x[ix]
            actions=[old@initial_readout,new@initial_readout,old@wf,new@w,
                     (new@w)*scale,(new@initial_readout)/scale]
            def metrics(ids):
                return trial.scoring.audit.blockstats(p[evaluate[ids]],y[evaluate[ids]],[v[ids] for v in actions]).sum(0)
            sums[j].append(metrics(np.arange(len(ix))))
            if not j:
                parts=[evaluate<2300,evaluate>=2460]
                regions.append([trial.scoring.audit.pooled(metrics(take)).tolist() for take in parts])
        rolling_rows.append(tuple(np.concatenate(v) for v in zip(*fresh_records)))
        fixed_rows.append(tuple(np.concatenate(v) for v in zip(*fixed_records)))
        if d>=8 and d%2:
            folds.append([trial.scoring.audit.pooled(np.sum(group,0)).tolist() for group in sums])
    folds=np.array(folds)
    reference=json.loads((model.OUT/'results.json').read_text())
    delta=folds[:,:,3]-np.array(reference['folds'])[:,:,0,7]
    assert abs(delta).max()<1e-7
    report=dict(days=list(range(9,17)),arms=['fixed at day eight','refit modes only','refit Y3 readout only','refit both','new direction with earlier magnitude','earlier direction with new magnitude'],
                magnitude_ratio_by_fold=scale_factors,
                folds=folds.tolist(),sessions=['before tick 2300','from tick 2460'],daily_session_gains=regions,
                mean=folds.mean(0).tolist(),post_first_fold_mean=folds[1:].mean(0).tolist())
    (model.OUT/'refit_audit.json').write_text(json.dumps(report,indent=2))
    print('mean Y3 gains for days 11–16',np.round(report['post_first_fold_mean'],6),flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
