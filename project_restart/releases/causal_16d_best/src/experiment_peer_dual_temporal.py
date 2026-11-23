"""Frozen equal TCN/LRU prediction mixture, audited on earlier folds only.

No fitting, weight search, GPU operation, or late-label access occurs here.
The two independently trained encoders and all existing output routing remain
unchanged; the sole candidate is 0.5 TCN + 0.5 LRU in original target units.
"""
import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

import experiment_peer_temporal_arch as original

common = original.common
scales = original.scales
peer = original.peer
graphs = original.graphs
norm = original.norm
outer = original.outer
components = original.components
routing = original.routing
OUT = common.ROOT / 'artifacts/project_restart_generalization/peer_dual_temporal'
BEST = original.OUT / 'commit_peer_temporal'
MODE = 'fixed_equal'
MODES = ('tcn_local', 'complex_lru')
WEIGHTS = (0.5, 0.5)
build = original.build
predict = original.predict


def read_json(path):
    return json.loads(Path(path).read_text())


def verify_best():
    report = read_json(original.OUT / 'official_results.json')
    if report.get('promoted') is not True or Path(report['commit']).resolve() != BEST.resolve():
        raise RuntimeError('Current local TCN package is not the promoted baseline')
    actual = {name: common.sha256(BEST / name) for name in report['hashes']}
    if len(actual) != 8 or actual != report['hashes']:
        raise RuntimeError('Current TCN package changed')
    return actual


def verify_original_plan():
    path = original.OUT / 'plan.json'
    plan = read_json(path)
    expected = {'seed': 42, 'epochs': 2, 'window': 16, 'row_batch': 1024,
                'lr': 1e-4, 'weight_decay': 1e-4, 'clip': 1., 'main_weights': [.375, .625]}
    if plan['folds'] != common.FOLDS or plan['data_sha256'] != common.EXPECTED_SHA or plan['training'] != expected:
        raise RuntimeError('Original temporal training plan differs')
    if any(common.sha256(name) != sha for name, sha in plan['source_sha256'].items()):
        raise RuntimeError('An original frozen source changed')
    return plan


def graph_for(tag):
    if tag == 'outer':
        path = graphs.OUT / 'outer_graph.npy'
        graph = np.asarray(np.load(path), dtype=np.float32)
    elif tag in common.FOLDS:
        path = graphs.OUT / f'graph_{tag}.npz'
        with np.load(path) as saved:
            graph = np.asarray(saved['target_graph'], dtype=np.float32)
    else:
        raise ValueError(tag)
    if graph.shape != (50, 50) or not np.isfinite(graph).all():
        raise RuntimeError('Invalid peer graph')
    np.testing.assert_allclose(graph.sum(1), 1., rtol=0, atol=1e-6)
    return graph, path


def row_count(days):
    # Only read early/training X headers; never infer sizes from late labels.
    if any(int(day) not in range(16) for day in days):
        raise ValueError('This pilot only permits dates 0 through 15')
    return sum(len(np.load(norm.CACHE / f'day_{day:02d}_x.npy', mmap_mode='r')) * 50 for day in days)


def verify_source(mode, tag, seed):
    if mode not in MODES or tag not in ('A', 'B', 'outer') or seed not in (42, 43, 44):
        raise ValueError('Invalid source selection')
    if tag != 'outer' and seed != 42:
        raise ValueError('Early controls only use seed 42')
    verify_original_plan()
    directory = OUT if mode == 'complex_lru' and tag == 'outer' else original.OUT
    path = directory / f'{mode}_{tag}_s{seed}.pth'
    metadata_path = path.with_suffix('.json')
    metadata = read_json(metadata_path)
    graph, graph_path = graph_for(tag)
    graph_sha = hashlib.sha256(graph.tobytes()).hexdigest()
    days = list(range(16)) if tag == 'outer' else common.FOLDS[tag]['train']
    rows = row_count(days)
    expected = {'mode': mode, 'seed': seed, 'rows': rows, 'window': 16, 'epochs': 2,
                'row_batch': 1024, 'plan_sha256': common.sha256(original.OUT / 'plan.json'),
                'graph_sha256': graph_sha}
    if metadata['fingerprint'] != expected or len(metadata['records']) != 2:
        raise RuntimeError('Original model fingerprint or epochs differ')
    batches = (rows + 1023) // 1024
    for record in metadata['records']:
        if record['rows'] != rows or record['batches'] != batches:
            raise RuntimeError('Source training coverage is incomplete')
        if not np.isfinite([record['weighted_loss'], *record['head_mse']]).all():
            raise RuntimeError('Nonfinite source training losses')
    parameter_count = {'tcn_local': 203650, 'complex_lru': 203970}[mode]
    if metadata['parameters'] != parameter_count:
        raise RuntimeError('Source architecture size differs')
    weight_mean = metadata['weight_mean']
    if not np.isfinite(weight_mean) or weight_mean <= 0 or common.sha256(path) != metadata['sha256']:
        raise RuntimeError('Source checkpoint hash or scale normalization differs')
    return {'mode': mode, 'tag': tag, 'seed': seed, 'checkpoint': str(path),
            'checkpoint_sha256': metadata['sha256'], 'metadata': str(metadata_path),
            'metadata_sha256': common.sha256(metadata_path), 'fingerprint': expected,
            'rows': rows, 'weight_mean': weight_mean, 'graph': str(graph_path),
            'graph_sha256': graph_sha, 'graph_file_sha256': common.sha256(graph_path),
            'plan_sha256': expected['plan_sha256']}


def validate_comparison(record, days):
    if set(record['daily']) != {str(day) for day in days}:
        raise RuntimeError('Incorrect or incomplete early dates')
    for values in (record, *record['daily'].values()):
        triple = [values['base'], values['candidate'], values['delta']]
        if not np.isfinite(triple).all() or abs(triple[1] - triple[0] - triple[2]) > 1e-10:
            raise RuntimeError('Invalid comparison metrics')


def eligibility(folds):
    if set(folds) != set(common.FOLDS):
        raise RuntimeError('Both complete early folds are required')
    for fold, spec in common.FOLDS.items():
        validate_comparison(folds[fold], spec['validation'])
    positive = sum(day['delta'] > 0 for record in folds.values() for day in record['daily'].values())
    minimum = min(record['delta'] for record in folds.values())
    return {'passed': bool(minimum >= .001 and positive >= 6),
            'daily_positive': int(positive), 'minimum_gain_vs_tcn': float(minimum)}


def frozen_plan():
    previous = verify_original_plan()
    files = set(previous['source_sha256']) | {str(Path(__file__).resolve())}
    cache_files = []
    for fold, spec in common.FOLDS.items():
        for mode in MODES:
            cache_files.extend(original.OUT / f'{mode}_{fold}_{ending}.npy' for ending in ('heads', 'prediction'))
        cache_files.extend(norm.CACHE / f'day_{day:02d}_y.npy' for day in spec['validation'])
    return {'mode': MODE, 'weights': list(WEIGHTS), 'folds': common.FOLDS,
            'data_sha256': common.EXPECTED_SHA,
            'source_sha256': {name: common.sha256(name) for name in sorted(files)},
            'original_plan_sha256': common.sha256(original.OUT / 'plan.json'),
            'source_checkpoints': {fold: {mode: verify_source(mode, fold, 42) for mode in MODES} for fold in common.FOLDS},
            'outer_tcn_checkpoints': {str(seed): verify_source('tcn_local', 'outer', seed) for seed in (42, 43, 44)},
            'cache_sha256': {str(path): common.sha256(path) for path in cache_files},
            'baseline': str(BEST), 'best_hashes': verify_best(),
            'baseline_report_sha256': common.sha256(original.OUT / 'official_results.json'),
            'combination': 'q=0.5*TCN_scaled_head2+0.5*LRU_scaled_head2; original routing(p1,q)',
            'training': 'No early training; reuse audited seed42 E2 original independently trained encoders',
            'frozen': 'No coefficient search or fine-tuning; p0/p1, graph, scale, normalization and routing unchanged',
            'gate': 'Gain >=.001 vs current localTCN on each early fold, >=6/8 positive daily gains',
            'late_dates': 'Never read by this pilot; any later evaluation is retrospective, not a blind holdout',
            'prior_evidence': 'Equal-weight complementarity already inspected on these early folds; this is provenance verification, not a new independent validation'}


def verify_gate():
    """Recompute the gate and provenance before any caller loads full/late data."""
    plan = read_json(OUT / 'plan.json')
    if frozen_plan() != plan:
        raise RuntimeError('Frozen dual-temporal plan or sources changed')
    early = read_json(OUT / 'pilot_results.json')
    if early.get('plan_sha256') != common.sha256(OUT / 'plan.json') or early.get('late_dates_evaluated') is not False:
        raise RuntimeError('Pilot plan or date boundary differs')
    status = eligibility(early['folds'])
    if not status['passed'] or early.get('selected') != MODE or early.get('eligibility', {}).get(MODE) != status:
        raise RuntimeError('Fixed equal candidate did not pass both early folds')
    if early.get('best_hashes') != verify_best():
        raise RuntimeError('Pilot baseline package differs')
    for path, expected in early['output_sha256'].items():
        if common.sha256(path) != expected:
            raise RuntimeError('An audited early prediction changed')
    return early, plan


def cached_control(src, X, tid, scale, p1, fold, mode, source):
    heads_path = original.OUT / f'{mode}_{fold}_heads.npy'
    final_path = original.OUT / f'{mode}_{fold}_prediction.npy'
    heads = np.load(heads_path)
    final = np.load(final_path)
    if heads.shape != (len(X), 2) or final.shape != (len(X),):
        raise RuntimeError('Cached prediction shape differs')
    if not np.isfinite(heads).all() or not np.isfinite(final).all():
        raise RuntimeError('Cached predictions contain nonfinite values')
    route_error = float(np.max(np.abs(routing.routed(p1, heads[:, 1]) - final)))
    if route_error > 1e-6:
        raise RuntimeError('Cached final prediction does not match current routing')
    starts = np.flatnonzero(tid[::50] == 0) * 50
    offsets = np.array([0, 49, 50, 15 * 50, 16 * 50])
    indices = np.unique(np.concatenate([np.arange(50), (starts[:, None] + offsets).ravel(),
                                         np.linspace(0, len(X) - 1, 40).astype(int)]))
    model = build(mode).cpu().eval()
    state = torch.load(source['checkpoint'], map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    with torch.no_grad():
        actual = model(torch.from_numpy(src.make_batch(X, indices, tid, 16)).float()).numpy() * scale[indices, None]
    head_error = float(np.max(np.abs(actual - heads[indices])))
    if not np.isfinite(head_error) or head_error > 5e-5:
        raise RuntimeError('CPU spread prediction differs from cached GPU heads')
    check = {'source': source, 'head_prediction_sha256': common.sha256(heads_path),
             'final_prediction_sha256': common.sha256(final_path),
             'cpu_spread_rows': len(indices), 'max_abs_heads': head_error,
             'max_abs_final': route_error}
    del model, state
    return heads, final, check


def save_array(path, array):
    temporary = path.with_suffix('.pending.npy')
    with temporary.open('wb') as stream:
        np.save(stream, array)
    temporary.replace(path)


def pilot(src):
    if common.sha256(common.DATA) != common.EXPECTED_SHA:
        raise RuntimeError('Original data hash differs')
    plan = frozen_plan()
    OUT.mkdir(parents=True, exist_ok=True)
    plan_path = OUT / 'plan.json'
    if plan_path.exists() and read_json(plan_path) != plan:
        raise RuntimeError('Existing output belongs to a different plan')
    common.save_json(plan_path, plan)
    result = {'folds': {}, 'controls': {}, 'selected': None, 'late_dates_evaluated': False,
              'plan_sha256': common.sha256(plan_path), 'output_sha256': {}}
    reference = scales.reference()
    for fold, spec in common.FOLDS.items():
        print(json.dumps({'stage': 'cpu_cached_audit', 'fold': fold}), flush=True)
        raw, target, tid = norm.load_days(spec['validation'], normalized=True)
        raw = raw.astype(np.float32)
        graph, _ = graph_for(fold)
        X = peer.PeerArray(raw, graphs.context(raw, graph))
        scale = scales.load_scale(spec['validation'], reference)
        p1, _, _ = components.early_components(fold, spec['validation'])
        if p1.shape != (len(X),) or not np.isfinite(p1).all():
            raise RuntimeError('Current frozen p1 is invalid')
        controls = {}
        heads, finals = {}, {}
        for mode in MODES:
            heads[mode], finals[mode], controls[mode] = cached_control(
                src, X, tid, scale, p1, fold, mode, plan['source_checkpoints'][fold][mode])
        mixed_heads = WEIGHTS[0] * heads['tcn_local'] + WEIGHTS[1] * heads['complex_lru']
        mixed = routing.routed(p1, mixed_heads[:, 1])
        np.testing.assert_allclose(mixed, .5 * finals['tcn_local'] + .5 * finals['complex_lru'], rtol=0, atol=1e-6)
        record = scales.evaluate_prediction(mixed, finals['tcn_local'], target, spec['validation'], scale)
        record['head_corr'] = norm.corr(mixed_heads[:, 1], target[:, 2])
        record['head_std'] = float(mixed_heads[:, 1].std())
        record['head_mse'] = float(np.mean((mixed_heads[:, 1].astype(np.float64) - target[:, 2]) ** 2))
        validate_comparison(record, spec['validation'])
        result['folds'][fold] = record
        result['controls'][fold] = controls
        for ending, array in (('heads', mixed_heads), ('prediction', mixed)):
            path = OUT / f'{ending}_{fold}.npy'
            save_array(path, array)
            result['output_sha256'][str(path)] = common.sha256(path)
        common.save_json(OUT / 'pilot_partial.json', result)
        print(json.dumps({'stage': 'fold_complete', 'fold': fold, 'corr': record['candidate'],
                          'gain_vs_tcn': record['delta']}), flush=True)
        del raw, target, tid, X, scale, p1, heads, finals, mixed_heads, mixed
        gc.collect()
    status = eligibility(result['folds'])
    result['eligibility'] = {MODE: status}
    result['selected'] = MODE if status['passed'] else None
    result['best_hashes'] = verify_best()
    result['best_hashes_unchanged'] = result['best_hashes']
    common.save_json(OUT / 'pilot_results.json', result)
    print(json.dumps({'stage': 'pilot_complete', 'selected': result['selected'], 'eligibility': status}), flush=True)
    return result


def preflight():
    rng = np.random.default_rng(41)
    p1, tcn, lru = [rng.normal(size=400).astype(np.float32) for _ in range(3)]
    np.testing.assert_allclose(routing.routed(p1, .5 * tcn + .5 * lru),
                               .5 * routing.routed(p1, tcn) + .5 * routing.routed(p1, lru), rtol=0, atol=1e-6)
    records = {}
    for fold, spec in common.FOLDS.items():
        records[fold] = {'base': .2, 'candidate': .202, 'delta': .002,
                         'daily': {str(day): {'base': .2, 'candidate': .202, 'delta': .002} for day in spec['validation']}}
    assert eligibility(records)['passed']
    records['A']['candidate'] = .2005
    records['A']['delta'] = .0005
    assert not eligibility(records)['passed']
    return {'routing_linear_in_frozen_equal_heads': True, 'gate_requires_both_folds': True,
            'weights': list(WEIGHTS), 'cpu_only': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=('preflight', 'pilot'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    tests = preflight()
    if args.stage == 'preflight':
        print(json.dumps(tests), flush=True)
        return
    src, _ = outer.prepare()
    torch.set_num_threads(4)
    pilot(src)


if __name__ == '__main__':
    main()
