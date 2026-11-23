"""Test an independently trained distributional, bid/ask, state-space and event-GRU portfolio."""
import json
import numpy as np
import torch
import experiment_asymmetric_return_distribution_v1 as dist
import experiment_bid_ask_side_network_v1 as side
import experiment_damped_market_state_v1 as damp
import experiment_quote_event_gru_v1 as event
import experiment_phase_future_book_v1 as phase
import experiment_consensus_horizon_portfolios_v1 as blend
import experiment_tail_portfolio_filter_v1 as panel
import experiment_seasonal_pressure_surprises_v1 as seasonal
import experiment_consensus_error_variance_v1 as variance
import experiment_matured_portfolio_posterior_v1 as online
from experiment_replicated_expert_consensus_v1 import sources
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'architecture_family_portfolio_v1'
MODELS=(dist,side,damp,event)
FRACTION=.15


def loader(module,end):
    checkpoint=module.OUT/f'net_{end}.pth'
    if not checkpoint.exists():
        if end!=8:raise FileNotFoundError(checkpoint)
        net=module.train(8)
    else:
        st=torch.load(checkpoint,map_location='cpu',weights_only=True)
        if module is dist:net=dist.DistributionNet(st['current.0.weight'].shape[1])
        elif module is side:net=side.SideNet(st['encoder.0.weight'].shape[1])
        elif module is damp:net=damp.restore(st)
        else:net=event.EventNet(109)
        net.load_state_dict(st);net.cuda().eval()
    fn=phase.infer if module is dist else module.infer
    return lambda raw:fn(net,raw)


def side_path(module,d):
    existing=module.OUT/f'side_{d}.npy'
    return existing if existing.is_file() else OUT/f'{module.OUT.name}_{d}.npy'


def prepare():
    for module in MODELS:
        missing=[d for d in range(8,12) if not side_path(module,d).is_file()]
        if not missing:continue
        fn=loader(module,8)
        for d in missing:
            raw=phase.inn.grid(d);pred=fn(raw);short=fn(raw[:249])
            if not np.isfinite(pred).all() or not np.allclose(pred[:249],short,rtol=0,atol=.0001):
                raise ValueError(f'{module.OUT.name} early prefix mismatch')
            path=side_path(module,d);temp=path.with_suffix('.tmp')
            with temp.open('wb') as f:np.save(f,pred)
            temp.replace(path)
        del fn;torch.cuda.empty_cache()


def delta(p,forecasts):
    x=np.mean(forecasts,axis=0)-p
    x[:,:,2]-=panel.RATIO*x[:,:,1]
    return x-x.mean(1,keepdims=True)


def current_model():
    fixed=np.load(blend.PARENT/'weights.npy')[0];ow=np.load(online.OUT/'weights.npy')
    with np.load(variance.OUT/'weights.npz') as st:initial,cw=st['prior'],st['w']
    with np.load(seasonal.OUT/'model.npz') as st:season,w=(st['mean'],st['scale']),st['w']
    def predict(raw,p,fs):
        p=blend.parent.fixed_forecast(p,fs,fixed);p=online.advance(p,online.components(raw,p,fs),ow)
        p=(p+variance.features(fs,initial)@cw).astype('float32')
        return seasonal.apply(p,seasonal.features(raw,p,season,True),w)
    return predict


def verify(gate):
    import polars as pl
    import replay_session_experts_16d as replay
    import experiment_peer_implied_tail_v1 as peer
    from experiment_peer_quote_reliability_v1 import partial
    extra=[loader(m,16) for m in MODELS];old=sources();current=current_model()
    ref=json.loads((seasonal.OUT/'results.json').read_text())['development']
    with np.load(peer.OUT/'model.npz') as st:graph,coef=st['peers'],st['readout']
    def predict(raw,p):
        a=current(raw,partial(raw,p,graph,coef),[fn(raw) for fn in old])
        q=(a+FRACTION*delta(a,[fn(raw) for fn in extra])*gate).astype('float32')
        return a,q
    rows=[];daily={}
    for d in range(16,20):
        raw=replay.features(replay.DATA,d);p,head,n=panel.replay_parent(d,raw)
        prior,q=predict(raw,p);_,short=predict(raw[:n],head)
        err=float(abs(q[:n]-short).max())
        if not np.isfinite(q).all() or err>=1e-4:raise ValueError('invalid prefix')
        y=pl.scan_parquet(replay.DATA).filter(pl.col('date')==d).sort('time','code').select('y0','y1','y2').collect().to_numpy()
        row=prior.reshape(-1,3),q.reshape(-1,3),y;rows.append(row);s=measure([row])
        if abs(s['baseline_score']-ref['daily'][str(d+1)]['score'])>2e-6:raise ValueError('incorrect parent')
        daily[d+1]=dict(score=s['score'],gain=s['gain'],prefix_error=err)
    s=measure(rows)
    if s['rows']!=936250:raise ValueError('incorrect row count')
    return dict(pooled=s,daily=daily)


def main():
    OUT.mkdir(exist_ok=True);prepare();history=[];report={'folds':[]};daily=[]
    for end,stop in ((8,12),(12,14),(14,16)):
        c=blend.fixed_weight(history);rows=[]
        for d in range(end,stop):
            p,y,fs,stat=blend.training_day(d,history);history.append(stat)
            p=blend.parent.fixed_forecast(p,fs,c)
            z=delta(p,[np.load(side_path(m,d),mmap_mode='r') for m in MODELS])
            q=(p+FRACTION*z).astype('float32')
            row=tuple(v.reshape(-1,3) for v in (p,q,y));rows.append(row)
            s=measure([row]);daily.append(np.subtract(s['candidate'],s['base']))
        score=measure(rows);report['folds'].append(score);print(end+1,score,flush=True)
    gains=np.stack([np.subtract(f['candidate'],f['base']) for f in report['folds']])
    gate=(gains.min(0)>0)&((np.stack(daily)>0).sum(0)>=5)
    report.update(retained=gate.tolist(),fraction=FRACTION,sources=[m.OUT.name for m in MODELS])
    path=OUT/'results.json';path.write_text(json.dumps(report,indent=2))
    if gate.any():
        np.save(OUT/'gate.npy',gate)
        report['development']=verify(gate);path.write_text(json.dumps(report,indent=2));print(report['development'],flush=True)


if __name__=='__main__':main()
