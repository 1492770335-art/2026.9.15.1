"""Rebuild the release from parquet features and both trained model stages."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import polars as pl

import replay_session_experts_16d as session


STAGES = ('common', 'portfolios', 'rotation', 'joint', 'peers', 'settled', 'attention', 'area', 'pressure', 'participation', 'echo-vintages', 'pair-alignment', 'early-stock-slopes', 'supported-morning', 'directed-prices', 'reversal-memory', 'bipower', 'phase-pool', 'absolute-motion', 'pressure-cohort', 'peer-trajectory', 'magnitude-ratio', 'expected-occupancy', 'revision-feedback', 'medium-feedback', 'short-price-feedback', 'variance-ratio-magnitude', 'short-medium-ratio', 'market-disagreement-gate', 'vintage-orientation', 'partial-maturity', 'revision-transport', 'peer-sensors', 'fast-peer', 'peer-settlement', 'matchup', 'medium-matchup')

def replay(folder, data, stage="common"):
    import torch
    from experiment_causal_cross_horizon_ridge_v1 import measure
    import experiment_y3_session_continuation_ridge_v1 as common
    import experiment_y3_session_characteristic_portfolios_v1 as portfolios
    import experiment_y3_realized_sort_rotation_v1 as rotation
    import experiment_joint_sort_horizon_transfer_v1 as joint
    import experiment_y3_peer_response_fingerprints_v1 as peers
    import experiment_y3_settled_stock_gradient_v1 as settled
    import experiment_y3_settled_error_attention_v1 as memory
    import experiment_y3_ordered_book_area_v1 as area
    import experiment_y3_unpriced_peer_pressure_v1 as pressure
    import experiment_y3_peer_shock_participation_v1 as participation
    import experiment_y3_echo_vintage_ensemble_v1 as echoes
    import experiment_y3_temporal_pair_alignment_v1 as alignment
    import experiment_y3_early_settled_stock_slopes_v1 as stock_slopes
    import experiment_y3_supported_morning_slopes_v1 as supported
    import experiment_y3_directed_unexpected_prices_v1 as directed_prices
    import experiment_y3_reversal_conditioned_memory_v1 as reversal
    import experiment_y3_local_jump_variation_v1 as bipower
    import experiment_y3_bipower_phase_pool_v1 as phase_pool
    import experiment_y3_absolute_block_motion_v1 as motion
    import experiment_y3_pressure_cohort_projection_v1 as cohort
    import experiment_y3_forecast_trajectory_coupling_v1 as trajectory
    import experiment_y3_incremental_magnitude_ratio_v1 as magnitude
    import experiment_y3_expected_volatility_occupancy_v1 as occupancy
    import experiment_y3_pooled_revision_feedback_v1 as feedback
    import experiment_y2_matured_feedback_readout_v1 as medium_feedback
    import experiment_y1_price_feedback_v1 as short_feedback
    import experiment_y3_variance_ratio_magnitude_v1 as ratio_magnitude
    import experiment_short_medium_variance_ratio_v1 as short_medium_ratio
    import experiment_y3_prequential_disagreement_gate_v1 as market_gate
    import experiment_y3_intraday_vintage_orientation_v1 as orientation
    import experiment_y3_partial_maturity_orientation_v1 as partial_maturity
    import experiment_y3_matured_revision_factor_transport_v1 as revision_transport
    import experiment_y3_independent_peer_sensors_v1 as peer_sensors
    import experiment_y1_source_specific_peer_response_v1 as fast_peer
    import experiment_y3_peer_settled_medium_errors_v1 as peer_settlement
    import experiment_y3_forecast_matchup_difficulty_v1 as matchup
    import experiment_y2_whole_forecast_matchups_v1 as medium_matchup
    from released_model import SessionRefinement

    torch.set_num_threads(2)
    print('loading refinement parameters', flush=True)
    stage_index = STAGES.index(stage)
    module = {'common': common, 'portfolios': portfolios}.get(stage, rotation)
    model = SessionRefinement(
        update_factory=module.correction,
        include_medium=stage_index >= STAGES.index('joint'),
        include_peer_response=stage_index >= STAGES.index('peers'),
        include_settled=stage_index >= STAGES.index('settled'),
        include_attention=stage_index >= STAGES.index('attention'),
        include_ordered_area=stage_index >= STAGES.index('area'),
        include_peer_pressure=stage_index >= STAGES.index('pressure'),
        include_participation=stage_index >= STAGES.index('participation'),
        include_echo_vintages=stage_index >= STAGES.index('echo-vintages'),
        include_pair_alignment=stage_index >= STAGES.index('pair-alignment'),
        include_early_stock_slopes=stage_index >= STAGES.index('early-stock-slopes'),
        include_supported_morning=stage_index >= STAGES.index('supported-morning'),
        include_directed_prices=stage_index >= STAGES.index('directed-prices'),
        include_reversal=stage_index >= STAGES.index('reversal-memory'),
        include_bipower=stage_index >= STAGES.index('bipower'),
        include_phase_pool=stage_index >= STAGES.index('phase-pool'),
        include_absolute_motion=stage_index >= STAGES.index('absolute-motion'),
        include_pressure_cohort=stage_index >= STAGES.index('pressure-cohort'),
        include_peer_trajectory=stage_index >= STAGES.index('peer-trajectory'),
        include_magnitude_ratio=stage_index >= STAGES.index('magnitude-ratio'),
        include_expected_occupancy=stage_index >= STAGES.index('expected-occupancy'),
        include_revision_feedback=stage_index >= STAGES.index('revision-feedback'),
        include_medium_feedback=stage_index >= STAGES.index('medium-feedback'),
        include_short_feedback=stage_index >= STAGES.index('short-price-feedback'),
        include_ratio_magnitude=stage_index >= STAGES.index('variance-ratio-magnitude'),
        include_short_medium_ratio=stage_index >= STAGES.index('short-medium-ratio'),
        include_market_gate=stage_index >= STAGES.index('market-disagreement-gate'),
        include_vintage_orientation=stage_index >= STAGES.index('vintage-orientation'),
        include_partial_maturity=stage_index >= STAGES.index('partial-maturity'),
        include_revision_transport=stage_index >= STAGES.index('revision-transport'),
        include_peer_sensors=stage_index >= STAGES.index('peer-sensors'),
        include_fast_peer=stage_index >= STAGES.index('fast-peer'),
        include_peer_settlement=stage_index >= STAGES.index('peer-settlement'),
        include_matchup=stage_index >= STAGES.index('matchup'),
        include_medium_matchup=stage_index >= STAGES.index('medium-matchup'),
    )
    report_dir = {
        'joint': joint.OUT, 'peers': peers.OUT, 'settled': settled.PRIMARY_OUT,
        'attention': memory.OUT, 'area': area.OUT, 'pressure': pressure.OUT,
        'participation': participation.OUT, 'echo-vintages': echoes.OUT, 'pair-alignment': alignment.OUT, 'early-stock-slopes': stock_slopes.OUT, 'supported-morning': supported.OUT, 'directed-prices': directed_prices.OUT, 'reversal-memory': reversal.OUT, 'bipower': bipower.OUT, 'phase-pool': phase_pool.OUT, 'absolute-motion': motion.OUT, 'pressure-cohort': cohort.OUT, 'peer-trajectory': trajectory.OUT, 'magnitude-ratio': magnitude.OUT, 'expected-occupancy': occupancy.OUT, 'revision-feedback': feedback.OUT, 'medium-feedback': medium_feedback.OUT, 'short-price-feedback': short_feedback.OUT, 'variance-ratio-magnitude': ratio_magnitude.OUT, 'short-medium-ratio': short_medium_ratio.OUT, 'market-disagreement-gate': market_gate.OUT, 'vintage-orientation': orientation.OUT, 'partial-maturity': partial_maturity.OUT, 'revision-transport': revision_transport.OUT, 'peer-sensors': peer_sensors.OUT, 'fast-peer': fast_peer.OUT, 'peer-settlement': peer_settlement.OUT, 'matchup': matchup.OUT, 'medium-matchup': medium_matchup.OUT,
    }.get(stage, module.OUT)
    expected = json.loads((report_dir / 'reload_verification.json').read_text())
    daily, rows = {}, []
    for day, ticks in session.PREFIX.items():
        raw = session.features(folder / 'observations.parquet', day)
        print(f'day {day+1} predicting all {len(raw)} ticks', flush=True)
        with np.load(folder / f'base_{day}.npz') as base:
            full = model.predict(raw, base['full'], base['opening'])
            print(f'day {day+1} checking the first {ticks} ticks', flush=True)
            head = model.predict(raw[:ticks], base['prefix'], base['opening_prefix'])
        error = float(np.max(abs(full[:ticks]-head)))
        labels = (pl.scan_parquet(data).filter(pl.col('date') == day).sort('time','code')
                  .select('y0','y1','y2').collect().to_numpy())
        prediction = full.reshape(-1,3)
        row = prediction, prediction, labels
        score = measure([row])['score']
        difference = score-expected['daily'][str(day+1)]['score']
        if error >= 1e-4 or abs(difference) >= 2e-7 or not np.isfinite(full).all():
            raise ValueError((day+1, error, difference))
        rows.append(row)
        daily[day+1] = dict(score=score, prefix_error=error, staged_score_difference=difference)
        print('verified packaged day', day+1, daily[day+1], flush=True)
    pooled = measure(rows)
    assert pooled['rows'] == 936250
    np.testing.assert_allclose(pooled['candidate'],expected['pooled']['candidate'],atol=2e-7,rtol=0)
    report = dict(pooled=pooled, daily=daily, input='feature-only parquet for both inference stages')
    (report_dir/'packaged_inference.json').write_text(json.dumps(report, indent=2))
    print('verified packaged score', pooled['score'], flush=True)


def main():
    if len(sys.argv)==4 and sys.argv[1]=='_base':
        session.original(Path(sys.argv[2]), Path(sys.argv[3]))
        return
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-src',type=Path,required=True,help='causal_16d/src containing the trained Predictor')
    parser.add_argument('--data',type=Path,default=session.DATA)
    parser.add_argument('--stage',choices=STAGES, default='common',
                        help='last refinement stage to replay')
    args=parser.parse_args()
    if not (args.base_src/'predictor.py').is_file():
        parser.error('missing predictor.py in --base-src')
    with tempfile.TemporaryDirectory(prefix='market-release-') as name:
        path=Path(name)
        (pl.scan_parquet(args.data).filter(pl.col('date').is_in(list(session.PREFIX)))
         .select('date','time','code',*session.FIELDS).sink_parquet(path/'observations.parquet'))
        subprocess.run([sys.executable,'-B',__file__,'_base',name,str(args.base_src.resolve())],check=True)
        replay(path,args.data,args.stage)


if __name__=='__main__':main()
