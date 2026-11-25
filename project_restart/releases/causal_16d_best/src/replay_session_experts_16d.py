"""End-to-end validation of the 16-day session and first-passage experts from raw ordered features."""
import json
import hashlib
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import polars as pl

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'project/data/public.parquet'
ART=ROOT/'artifacts/project_restart_generalization'
OUT=ART/'session_and_passage_calibrated_16d'
PREFIX={16:1337,17:2741,18:457,19:3199}
FIELDS=[f'f{j:03d}' for j in range(100)]


def features(path,d):
    frame=pl.scan_parquet(path).filter(pl.col('date')==d).select('time','code',*FIELDS).sort('time','code').collect()
    ticks=frame['time'].to_numpy().reshape(-1,50)
    codes=frame['code'].to_numpy().reshape(-1,50)
    if not ((ticks==np.arange(len(ticks))[:,None]).all() and (codes==np.arange(50)).all()):
        raise ValueError(f'unexpected stock grid on date {d}')
    return frame.select(FIELDS).to_numpy().astype(np.float16).reshape(-1,50,100)


def original(scratch, base_source=None):
    source = base_source or ROOT/'2026.9.15.1/project_restart/releases/causal_16d/src'
    sys.path.insert(0, str(source))
    sys.path.insert(1,str(ROOT/'.deps/tree_models'))
    from predictor import Predictor
    from replay_curve_quantile_market_v1 import opening
    model=Predictor()
    replay_opening=opening.replay_features
    captured=[]
    def capture(*args,**kwargs):
        result=replay_opening(*args,**kwargs)
        captured.append(np.asarray(result[0][1]).reshape(-1,50,3).copy())
        return result
    opening.replay_features=capture
    for d,n in PREFIX.items():
        raw=features(scratch/'observations.parquet',d)
        captured.clear()
        full=model.predict(raw)
        prefix=model.predict(raw[:n])
        if not np.allclose(full[:n],prefix,rtol=0,atol=.0001):
            raise ValueError(f'original prefix mismatch on date {d}')
        if len(captured)!=2:
            raise ValueError('missing original sequence forecasts')
        np.savez(scratch/f'base_{d}.npz',full=full,prefix=prefix,opening=captured[0],opening_prefix=captured[1])
        print('replayed base day',d+1,flush=True)


def additions():
    import torch
    import experiment_range_horizon_transfer_v1 as rng
    import experiment_directional_depth_tree_v1 as tree
    import experiment_checkpoint_disagreement_v1 as vintage
    import experiment_session_restart_v1 as restart
    import experiment_conditional_short_transfer_v1 as carry
    import experiment_first_passage_return_v1 as hit
    import experiment_replicated_revision_slope_v1 as slope
    state=torch.load(rng.expert.OUT/'net_16.pth',map_location='cpu',weights_only=True)
    expert=rng.expert.restore(state).eval()
    expert.load_state_dict(state)
    with np.load(rng.OUT/'weights.npz') as st:
        rw,transfer=st['weight'],st['transfer']
    books=[tree.pair.lgb.Booster(model_file=str(tree.OUT/f'band_{k}.txt')) for k in range(3)]
    tw=np.load(tree.OUT/'weights.npy');geometry=tree.geometry()
    nets=vintage.networks(16)
    with np.load(vintage.OUT/'readout.npz') as st:
        market,relative=st['market'],st['relative']
    with np.load(restart.OUT/'params.npz') as st:
        tick=int(st['boundary']);sw=st['weights']
    cw=np.load(carry.OUT/'coefficients.npy')
    state=torch.load(hit.OUT/'net_16.pth',map_location='cpu',weights_only=True)
    barrier=hit.Model(state['encoder.0.weight'].shape[1]).cuda().eval()
    barrier.load_state_dict(state)
    hw=np.load(hit.OUT/'coef.npy')
    calibration=np.load(slope.OUT/'weights.npy')
    def predict(raw,p,opening):
        p0=rng.forecast(p,rng.expert.infer(expert,raw),rw,transfer)
        p=(p0+tree.signal(raw,p0,books,geometry)@tw).astype('float32')
        m,r=vintage.features(raw,p,16,nets)
        p=(p+(m@market)[:,None]+r@relative).astype('float32')
        p,_=restart.advance(raw,p,tick,sw)
        q=carry.advance(raw,p,p0,cw)
        m,r=hit.readout.basis(p,hit.infer(raw,barrier))
        full=(q+(m@hw[0])[:,None]+r@hw[1]).astype('float32')
        return slope.predict(full,opening,calibration)
    return predict


def main(peer_tail=False, fixed_expert_tail=False, matured_experts=False, consensus_variance=False, seasonal_pressure=False):
    from replay_verified_forecast_parent_v1 import stats
    from replay_curve_quantile_market_v1 import summary
    output=OUT
    consensus_variance=consensus_variance or seasonal_pressure
    matured_experts=matured_experts or consensus_variance
    fixed_expert_tail=fixed_expert_tail or matured_experts
    peer_tail=peer_tail or fixed_expert_tail
    if peer_tail:
        import experiment_peer_implied_tail_v1 as peer
        output=peer.OUT
        prior=json.loads((OUT/'raw_replay.json').read_text())
        with np.load(output/'model.npz') as st:
            peers,weights=st['peers'],st['readout']
        def refine(raw,p):
            q=p.copy()
            q[:,:,2]+=sum(peer.changes(raw,p,peers,weights))
            return q
    if fixed_expert_tail:
        prior=json.loads((peer.OUT/'raw_replay.json').read_text())
        output=ART/'equal_expert_relative_tail_v1'
        fixed_output=output
    if matured_experts:
        import experiment_matured_portfolio_posterior_v1 as posterior
        prior=json.loads((fixed_output/'raw_replay.json').read_text())
        output=posterior.OUT
        online_weights=np.load(output/'weights.npy')
    if consensus_variance:
        import experiment_consensus_error_variance_v1 as consensus
        prior=json.loads((posterior.OUT/'raw_replay.json').read_text())
        output=consensus.OUT
        with np.load(output/'weights.npz') as params:
            consensus_weights,initial_cov=params['w'],params['prior']
    if seasonal_pressure:
        import experiment_seasonal_pressure_surprises_v1 as seasonal
        prior=json.loads((consensus.OUT/'raw_replay.json').read_text())
        output=seasonal.OUT
        with np.load(output/'model.npz') as params:
            season=(params['mean'],params['scale'])
            flow_weights=params['w']
    output.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output) as folder:
        scratch=Path(folder)
        obs=scratch/'observations.parquet'
        pl.scan_parquet(DATA).filter(pl.col('date').is_in(list(PREFIX))).select('date','time','code',*FIELDS).sink_parquet(obs)
        subprocess.run([sys.executable,'-B',__file__,'base',str(scratch)],check=True)
        predict=additions()
        if fixed_expert_tail:
            import experiment_matured_expert_tail_v1 as expert_tail
            from experiment_replicated_expert_consensus_v1 import sources
            experts=sources()
            blend=np.load(fixed_output/'weights.npy')[0]
            def add_experts(raw,p):
                forecasts=[fn(raw) for fn in experts]
                fixed=expert_tail.fixed_forecast(p,forecasts,blend)
                if matured_experts:
                    z=posterior.components(raw,fixed,forecasts)
                    online=posterior.advance(fixed,z,online_weights)
                    if consensus_variance:
                        change=consensus.features(forecasts,initial_cov)@consensus_weights
                        calibrated=(online+change).astype('float32')
                        if seasonal_pressure:
                            z=seasonal.features(raw,calibrated,season,True)
                            return seasonal.apply(calibrated,z,flow_weights),calibrated
                        return calibrated,online
                    return online,fixed
                return fixed,p
        daily={};tot=np.zeros((6,3));bt=tot.copy();pieces=[]
        previous=tot.copy()
        for d,n in PREFIX.items():
            raw=features(obs,d)
            with np.load(scratch/f'base_{d}.npz') as st:
                p=st['full'];q=predict(raw,p,st['opening'])
                head=predict(raw[:n],st['prefix'],st['opening_prefix'])
            if peer_tail:
                before=q
                q=refine(raw,before)
                head=refine(raw[:n],head)
            if fixed_expert_tail:
                q,before=add_experts(raw,q)
                head,_=add_experts(raw[:n],head)
            err=float(np.max(abs(q[:n]-head)))
            if not np.isfinite(q).all() or not err<1e-4:
                raise ValueError(f'invalid prediction or prefix on date {d}: {err}')
            y=pl.scan_parquet(DATA).filter(pl.col('date')==d).sort('time','code').select('y0','y1','y2').collect().to_numpy().astype(float)
            s=stats(q,y);t=stats(p,y);tot+=s;bt+=t;pieces.append((s,t))
            daily[d+1]={**summary(s),'gain':summary(s)['score']-summary(t)['score'],'prefix_ticks':n,'max_prefix_error':err}
            if peer_tail:
                old=stats(before,y)
                previous+=old
                if abs(summary(old)['score']-prior['daily'][str(d+1)]['score'])>2e-6:
                    raise ValueError('incumbent replay mismatch')
                daily[d+1]['gain_over_incumbent']=summary(s)['score']-summary(old)['score']
            if seasonal_pressure:
                np.savez(scratch/f'trace_{d}.npz',full=q,head=head,prefix_ticks=n,
                         raw_sha256=hashlib.sha256(np.ascontiguousarray(raw).view(np.uint8)).hexdigest())
            print('new model',d+1,daily[d+1],flush=True)
        if summary(tot)['rows']!=936250 or abs(summary(bt)['score']-.36034073798)>2e-6:
            raise ValueError('base replay or row count differs from released result')
        result={'training_days':[1,16],'validation_days':[17,20],'baseline':summary(bt),'pooled':summary(tot),'daily':daily,
                'leave_one_day_out_gains':[summary(tot-s)['score']-summary(bt-t)['score'] for s,t in pieces]}
        if peer_tail:
            record=json.loads((output/'results.json').read_text())
            if fixed_expert_tail:
                record=record['development']
            expected=record['pooled']['score']
            if abs(summary(tot)['score']-expected)>2e-6:
                raise ValueError('full and component replays disagree')
            result['incumbent']=summary(previous)
            result['gain_over_incumbent']=summary(tot)['score']-summary(previous)['score']
        if seasonal_pressure:
            trace=output/'raw_trace'
            trace.mkdir(exist_ok=True)
            for d in PREFIX:
                (scratch/f'trace_{d}.npz').replace(trace/f'day_{d}.npz')
        (output/'raw_replay.json').write_text(json.dumps(result,indent=2))
        print('pooled',result['pooled'],flush=True)


if __name__=='__main__':
    if len(sys.argv)>1 and sys.argv[1]=='base':
        original(Path(sys.argv[2]))
    elif sys.argv[1:] in ([],['--peer-tail'],['--fixed-expert-tail'],['--matured-experts'],['--consensus-variance'],['--seasonal-pressure']):
        main(peer_tail='--peer-tail' in sys.argv,fixed_expert_tail='--fixed-expert-tail' in sys.argv,
             matured_experts='--matured-experts' in sys.argv,consensus_variance='--consensus-variance' in sys.argv,
             seasonal_pressure='--seasonal-pressure' in sys.argv)
    else:
        raise SystemExit('usage: replay_session_experts_16d.py [--peer-tail | --fixed-expert-tail | --matured-experts | --consensus-variance | --seasonal-pressure]')
