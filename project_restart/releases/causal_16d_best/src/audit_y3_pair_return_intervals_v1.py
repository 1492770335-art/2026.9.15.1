"""Attribute a frozen paired-field forecast change to disjoint realized return intervals."""
import importlib
import json
import sys

import numpy as np
import torch

import experiment_compounded_return_audit_v1 as price
from experiment_y3_error_components_v1 import components
import y3_filter_validation as trial


def correlation(g):
    cov = g[1:, 1:] - np.outer(g[0, 1:], g[0, 1:]) / g[0, 0]
    return cov[0, 1:] / np.sqrt(cov[0, 0] * cov.diagonal()[1:])


def audit(name):
    module = importlib.import_module(name)
    report = json.loads((module.OUT / 'reload_verification.json').read_text())
    older = trial.prior.parent.correction()
    stages = [trial.prior.incumbent.correction(), trial.incumbent.correction()]
    ordered_parents = ('y3_interval_rank_heads_v1', 'y3_state_transition_sources_v1',
                       'y3_honest_pair_distribution_v1', 'y3_risk_rank_partition_v1',
                       'y3_pair_crossing_premia_v1', 'y3_two_sided_path_encoder_v1',
                       'y3_path_price_context_v1', 'y3_expanding_path_complement_v1',
                       'y3_scale_conditioned_path_v1', 'y3_path_magnitude_consensus_v1',
                       'y3_expected_price_paths_v1', 'y3_unfulfilled_price_moves_v1', 'y3_market_reversal_flow_v1',
                       'y3_overshoot_basket_agreement_v1', 'y3_monotone_horizon_projection_v1',
                       'y3_quiet_forecast_revisions_v1', 'y3_event_forecast_response_shape_v1',
                       'y3_same_deadline_event_curve_v1', 'y3_same_quote_memory_v1', 'y3_quote_event_tokens_v1', 'y3_inferred_print_impacts_v1',
                       'y3_trade_spell_survival_v1', 'y3_peer_depth_event_causes_v1', 'y3_peer_terminal_price_news_v1', 'y3_joint_order_price_rollout_v1', 'y3_persistent_order_modes_v1', 'y3_receiver_price_portfolios_v1', 'y3_realized_recipient_exposures_v1', 'y3_crossed_level_book_v1', 'y3_impact_mass_forecast_v1')
    if report['parent'] != trial.incumbent.OUT.name:
        if report['parent'] not in ordered_parents:
            raise ValueError(f"Unknown reference model {report['parent']}")
        for stage_name in ordered_parents[:ordered_parents.index(report['parent']) + 1]:
            stages.append(importlib.import_module('experiment_' + stage_name).correction())
    candidate = module.correction()
    k, coeff, missing = trial.curve.settled.historical_geometry()
    windows = [(0, 5), (5, 20), (20, 40), (40, 60)]
    grams = np.zeros((4, 4, 4))
    daily = {}
    actual, predictions = [], []
    for day in range(16, 20):
        raw = trial.scoring.trace.replay.original.features(trial.scoring.trace.replay.original.DATA, day)
        pred, _, _ = trial.scoring.trace.load_day(day, raw)
        pred[:, :, 2] += older(raw)
        for stage in stages:
            pred[:, :, 2] += stage(raw, pred)
        base = pred[:, :, 2].copy()
        pred[:, :, 2] += candidate(raw, pred)
        labels = trial.scoring.trace.replay.labels(day)
        original = pred.copy()
        original[:, :, 2] = base
        official = trial.scoring.trace.older.measure([(original.reshape(-1, 3), pred.reshape(-1, 3), labels)])
        assert abs(official['score'] - report['daily'][str(day + 1)]['score']) < 1e-10
        assert abs(official['gain'] - report['daily'][str(day + 1)]['gain']) < 1e-10
        actual.append(labels.reshape(pred.shape)[:, :, 1:].astype(float))
        predictions.append(pred[:, :, 1:].astype(float))
        path = price.compound(raw, k)
        bad = np.vstack((np.zeros((1, 50), int), np.cumsum(~price.present(raw, missing), axis=0)))
        ticks = np.arange(len(raw) - 60)
        valid = bad[ticks + 61] == bad[ticks]
        # At least two eligible stocks are needed for a same-tick relative return.
        count = valid.sum(1, keepdims=True)
        valid &= count >= 2

        def relative(panel):
            mean = (panel * valid).sum(1, keepdims=True) / count.clip(1)
            return (panel - mean)[valid]

        old, new = (relative(x[ticks].astype(float)) for x in (base, pred[:, :, 2]))
        changes = []
        for j, (start, stop) in enumerate(windows):
            realized = price.span(path, ticks + start, stop - start, k)
            design = np.stack((np.ones_like(old), relative(realized), old, new), axis=1)
            g = design.T @ design
            grams[j] += g
            a, b = correlation(g)
            changes.append(float(b - a))
        daily[day + 1] = dict(rows=len(old), correlation_changes=changes)
    pooled = [correlation(g) for g in grams]
    result = dict(windows=windows, rows=int(grams[0, 0, 0]), baseline=[float(v[0]) for v in pooled],
                  candidate=[float(v[1]) for v in pooled], change=[float(v[1] - v[0]) for v in pooled], daily=daily)
    y, p = np.concatenate(actual), np.concatenate(predictions)
    ratio = coeff[2, 1] / coeff[1, 1]
    result['all_label_rows'] = y.shape[0] * y.shape[1]
    result['official_horizon_components'] = {
        'y3': components(y[:, :, 1], p[:, :, 1]),
        'last40': components(y[:, :, 1] - ratio * y[:, :, 0], p[:, :, 1] - ratio * p[:, :, 0])
    }
    assert abs(result['official_horizon_components']['y3']['full_ic'] - report['pooled']['candidate'][2]) < 1e-9
    (module.OUT / 'disjoint_return_audit.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    audit(sys.argv[1])
