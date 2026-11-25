"""Evaluate late relative-return specialists on historical folds and verified 50-stock grids."""
import json

import numpy as np

import audit_y3_pressure_branches_v1 as trace
import experiment_y3_crossfit_transition_interaction_v1 as parent
import experiment_y3_residual_peer_curve_v1 as audit

clock, curve = parent.clock, parent.curve


def late(grid):
    _, coeff, _ = curve.settled.historical_geometry()
    return clock.center(grid[:,:,2] - float(coeff[2,1]/coeff[1,1])*grid[:,:,1])


def statistics(p,y,changes):
    full = audit.blockstats(p,y,changes)
    tail = np.zeros_like(p)
    truth = np.zeros_like(y)
    tail[:,:,2],truth[:,:,2] = late(p),late(y)
    return full,audit.blockstats(tail,truth,[clock.center(change) for change in changes])


def select(folds,daily):
    # dimensions: fold, reference model, outcome (Y3 or late relative), variant
    f=np.asarray(folds)
    d=np.asarray(daily)
    eligible=((f>0).sum(0).min((0,1))>=3)&(f.mean(0).min((0,1))>0)&(f[-1].min((0,1))>0)&((d>0).sum(0).min(0)>=5)
    ids=np.flatnonzero(eligible)
    return int(max(ids,key=lambda k:f[:,0,0,k].mean())) if len(ids) else None


def outer(module,reference=None,preceding=(),load_reference=None):
    out=module.OUT
    ref_name=parent.OUT.name if reference is None else reference.OUT.name
    source=parent.OUT/'composed_replay.json' if reference is None else reference.OUT/'reload_verification.json'
    expected=json.loads(source.read_text())
    comparison=None
    for name in ('raw_replay','reload_verification'):
        if load_reference is None:
            base_delta=parent.correction()
            chain=[stage.correction() for stage in preceding]
            if reference is not None:chain.append(reference.correction())
        update=module.correction()
        rows=[];daily={};blocks=[];late_blocks=[]
        for day in range(16,20):
            raw=trace.replay.original.features(trace.replay.original.DATA,day)
            labels=trace.replay.labels(day)
            if load_reference is None:
                base, prefix, n=trace.load_day(day,raw)
                base[:,:,2]+=base_delta(raw)
                prefix[:,:,2]+=base_delta(raw[:n])
                for previous in chain:
                    base[:,:,2]+=previous(raw,base)
                    prefix[:,:,2]+=previous(raw[:n],prefix)
            else:
                base,prefix,n=load_reference(day,raw)
            q=base.copy();q[:,:,2]+=update(raw,base)
            h=prefix.copy();h[:,:,2]+=update(raw[:n],prefix)
            err=float(np.max(np.abs(h-q[:n])))
            assert np.isfinite(q).all() and err<1e-4
            np.testing.assert_array_equal(q[:,:,:2],base[:,:,:2])
            row=(base.reshape(-1,3),q.reshape(-1,3),labels)
            s=trace.older.measure([row])
            assert abs(s['baseline_score']-expected['daily'][str(day+1)]['score'])<1e-10
            stats=statistics(base,labels.reshape(base.shape),[q[:,:,2].astype(float)-base[:,:,2]])
            blocks.append(stats[0]);late_blocks.append(stats[1])
            daily[str(day+1)]=dict(score=s['score'],gain=s['gain'],prefix_error=err,
                late_corr_gain=float(audit.pooled(stats[1].sum(0))[0]))
            rows.append(row)
        result=trace.older.measure(rows)
        assert result['rows']==936250
        np.testing.assert_allclose(result['base'],expected['pooled']['candidate'],rtol=0,atol=1e-10)
        report=dict(parent=ref_name,pooled=result,daily=daily,
            late_corr_gain=float(audit.pooled(sum(b.sum(0) for b in late_blocks))[0]),
            paired_score_percentiles=(.5*np.asarray(audit.intervals(blocks))[:,0]).tolist())
        if comparison:
            np.testing.assert_allclose(result['candidate'],comparison['pooled']['candidate'],atol=1e-10,rtol=0)
            for day in daily:
                assert abs(daily[day]['score']-comparison['daily'][day]['score'])<1e-10
        (out/(name+'.json')).write_text(json.dumps(report,indent=2))
        print(name,result['score'],result['gain'],report['late_corr_gain'],flush=True)
        comparison=report
        if result['gain']<=0:break


def verify_full_chain(module, on_day=None, expected_name='reload_verification.json'):
    expected=json.loads((module.OUT/expected_name).read_text())
    replay=trace.replay.evaluate(module.layer(),on_day=on_day)
    np.testing.assert_allclose(replay['pooled']['candidate'],expected['pooled']['candidate'],rtol=0,atol=1e-10)
    for day,daily in replay['daily'].items():
        assert abs(daily['score']-expected['daily'][day]['score'])<1e-10
    out=module.OUT/'full_chain_verification'
    out.mkdir(exist_ok=True)
    (out/'composed_replay.json').write_text(json.dumps(replay,indent=2))
    print(module.OUT.name,'full chain',replay['pooled']['score'],flush=True)


if __name__=='__main__':
    import importlib
    import sys
    import torch
    torch.set_num_threads(2)
    for name in sys.argv[1:]:
        verify_full_chain(importlib.import_module(name))
