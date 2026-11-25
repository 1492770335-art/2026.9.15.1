"""Replay the final heads from the independently verified upstream outputs and score in paired time blocks."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from audit_y3_persistent_exposure_transfer_v1 import gain, statistics
from experiment_causal_cross_horizon_ridge_v1 import measure
import experiment_y3_peer_response_fingerprints_v1 as peers
import experiment_joint_sort_horizon_transfer_v1 as joint
import experiment_y3_settled_stock_gradient_v1 as gradient
import experiment_y3_settled_error_attention_v1 as memory
import experiment_y3_ordered_book_area_v1 as area
import experiment_y3_unpriced_peer_pressure_v1 as pressure
import experiment_y3_peer_shock_participation_v1 as participation
import experiment_y3_echo_vintage_ensemble_v1 as echoes
import experiment_y3_temporal_pair_alignment_v1 as alignment
import experiment_y3_early_settled_stock_slopes_v1 as stock_slopes
import experiment_y3_supported_morning_slopes_v1 as supported
import experiment_y3_directed_unexpected_prices_v1 as price_response
import experiment_y3_reversal_conditioned_memory_v1 as reversal
import replay_session_experts_16d as session
from released_model import SessionRefinement


def audit(upstream_dir, data, settled=False, attention=False, since_peer=False, ordered_area=False, peer_pressure=False, concentration=False, since_area=False, echo_vintages=False, pair_alignment=False, early_stock_slopes=False, supported_morning=False, directed_prices=False, reversal_memory=False):
    import torch
    torch.set_num_threads(2)
    if reversal_memory:
        model = SessionRefinement(include_bipower=False)
        before = SessionRefinement(include_reversal=False)
        previous = price_response.OUT / 'packaged_heads.json'
        output = reversal.OUT
    elif directed_prices:
        model = SessionRefinement(include_reversal=False)
        before = SessionRefinement(include_directed_prices=False)
        previous = supported.OUT / 'packaged_heads.json'
        output = price_response.OUT
    elif supported_morning:
        model = SessionRefinement(include_directed_prices=False)
        before = SessionRefinement(include_supported_morning=False)
        previous = stock_slopes.OUT / 'packaged_heads.json'
        output = supported.OUT
    elif early_stock_slopes:
        model = SessionRefinement(include_supported_morning=False)
        before = SessionRefinement(include_early_stock_slopes=False)
        previous = alignment.OUT / 'packaged_heads.json'
        output = stock_slopes.OUT
    elif pair_alignment:
        model = SessionRefinement(include_early_stock_slopes=False)
        before = SessionRefinement(include_pair_alignment=False)
        previous = echoes.OUT / 'packaged_heads.json'
        output = alignment.OUT
    elif echo_vintages:
        model = SessionRefinement(include_pair_alignment=False)
        before = SessionRefinement(include_echo_vintages=False)
        previous = participation.OUT/'packaged_heads.json'
        output = echoes.OUT
    elif concentration:
        model=SessionRefinement(include_echo_vintages=False)
        before=SessionRefinement(include_participation=False,include_peer_pressure=not since_area)
        previous=(area.OUT if since_area else pressure.OUT)/'packaged_heads.json'
        output=participation.OUT
    elif peer_pressure:
        model=SessionRefinement(include_participation=False)
        before=SessionRefinement(include_peer_pressure=False)
        previous=area.OUT/'packaged_heads.json'
        output=pressure.OUT
    elif ordered_area:
        model=SessionRefinement(include_peer_pressure=False)
        before=SessionRefinement(include_ordered_area=False)
        previous=memory.OUT/'packaged_heads.json'
        output=area.OUT
    elif attention:
        model=SessionRefinement(include_ordered_area=False)
        before=SessionRefinement(include_settled=not since_peer,include_attention=False)
        previous=(peers.OUT if since_peer else gradient.PRIMARY_OUT)/'packaged_heads.json'
        output=memory.OUT
    else:
        model=SessionRefinement(include_settled=settled,include_attention=False)
        before=SessionRefinement(include_peer_response=settled, include_settled=False)
        previous=peers.OUT/'packaged_heads.json' if settled else joint.OUT/'packaged_inference.json'
        output=gradient.PRIMARY_OUT if settled else peers.OUT
    accepted=json.loads(previous.read_text())
    rows, groups, daily=[],[],{}
    stock_stats=np.zeros((50,4,4),dtype=float) if (settled or attention or ordered_area or peer_pressure or concentration or echo_vintages or pair_alignment or early_stock_slopes or supported_morning or directed_prices or reversal_memory) else None
    for d,n in session.PREFIX.items():
        raw=session.features(data,d)
        initial=np.load(upstream_dir/f'full_{d}.npy')
        old=before.finish_portfolios(raw,initial.copy())
        pred=model.finish_portfolios(raw,initial.copy())
        short=model.finish_portfolios(raw[:n],np.load(upstream_dir/f'prefix_{d}.npy'))
        error=float(abs(pred[:n]-short).max())
        assert error<1e-4 and np.isfinite(pred).all() and np.array_equal(old[:,:,:2],pred[:,:,:2])
        y=(pl.scan_parquet(data).filter(pl.col('date')==d).sort('time','code')
           .select('y0','y1','y2').collect().to_numpy())
        record=(old.reshape(-1,3),pred.reshape(-1,3),y)
        m=measure([record])
        assert abs(m['baseline_score']-accepted['daily'][str(d+1)]['score'])<1e-9
        rows.append(record)
        daily[d+1]=dict(score=m['score'],gain=m['gain'],prefix_error=error)
        delta=pred[:,:,2].astype(float)-old[:,:,2]
        yy=y.reshape(old.shape)
        if stock_stats is not None:
            vectors=np.stack((np.ones_like(delta),old[:,:,2],yy[:,:,2],delta),axis=2)
            stock_stats+=np.einsum('tsi,tsj->sij',vectors,vectors)
        blocks=[]
        for lo,hi in peers.parent.base.segments(len(raw)):
            for t in range(lo,hi,120):
                sl=slice(t,min(t+120,hi))
                blocks.append(statistics(old[sl],yy[sl],delta[sl]))
        groups.append(np.stack(blocks))
    result=measure(rows)
    assert result['rows']==936250
    sums=np.stack([g.sum(0) for g in groups])
    total=sums.sum(0)
    assert abs(float(gain(total))-result['gain'])<1e-10
    draws=[]
    rng=np.random.default_rng(82713)
    for _ in range(2500):
        sample=np.zeros_like(total)
        for d in rng.integers(4,size=4):
            group=groups[d]
            sample+=group[rng.integers(len(group),size=len(group))].sum(0)
        draws.append(float(gain(sample)))
    report=dict(pooled=result,daily=daily,rows=result['rows'],input='raw visible fields and frozen upstream forecasts',
                block_ticks=120,replicates=len(draws),omit_date={d+17:float(gain(total-s)) for d,s in enumerate(sums)},
                interval=np.quantile(draws,[.025,.975]).tolist())
    if stock_stats is not None:
        np.testing.assert_allclose(stock_stats.sum(0),total,atol=1.e-6,rtol=1.e-10)
        report['omit_stock']={i:float(gain(total-v)) for i,v in enumerate(stock_stats)}
        report['individual_stock_y3_gain']={i:float(2*gain(v)) for i,v in enumerate(stock_stats)}
    filename='paired_since_peer.json' if since_peer else 'packaged_heads.json'
    if since_area:
        filename='paired_since_area.json'
    (output/filename).write_text(json.dumps(report,indent=2))
    print('verified packaged head',result['score'],'CI',report['interval'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--upstream-dir',type=Path,required=True)
    parser.add_argument('--data',type=Path,default=session.DATA)
    parser.add_argument('--settled',action='store_true',help='compare the new settled-error model to the peer version')
    parser.add_argument('--attention',action='store_true',help='compare similarity-weighted retrieval to the settled gradient')
    parser.add_argument('--since-area',action='store_true',help='compare both peer-pressure layers to ordered book areas')
    parser.add_argument('--reversal-memory',action='store_true',help='test reversal-conditioned price memory')
    parser.add_argument('--directed-prices',action='store_true',help='test cross-stock responses to predictable price changes')
    parser.add_argument('--supported-morning', action='store_true', help='compare replicated exposure and morning transfer to early stock slopes')
    parser.add_argument('--early-stock-slopes', action='store_true', help='compare stock-specific short-interval response to lagged pairs')
    parser.add_argument('--pair-alignment', action='store_true', help='compare lagged stock pairs to the recurrent ensemble')
    parser.add_argument('--echo-vintages', action='store_true', help='compare recurrent peer states against the participation stage')
    parser.add_argument('--participation',action='store_true',help='compare shock concentration to persistent peer pressure')
    parser.add_argument('--pressure',action='store_true',help='compare persistent peer pressure to ordered book areas')
    parser.add_argument('--area',action='store_true',help='compare ordered book areas to the attention version')
    parser.add_argument('--since-peer',action='store_true',help='include the settled-gradient layer in the paired gain')
    args=parser.parse_args()
    if args.since_peer and not args.attention:
        parser.error('--since-peer requires --attention')
    if args.since_area and not args.participation:
        parser.error('--since-area requires --participation')
    audit(args.upstream_dir,args.data,args.settled,args.attention,args.since_peer,args.area,args.pressure,args.participation,args.since_area,args.echo_vintages,args.pair_alignment,args.early_stock_slopes,args.supported_morning,args.directed_prices,args.reversal_memory)
