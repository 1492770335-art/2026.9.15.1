"""Guarded full-prefix training and deployment for fixed equal TCN/LRU ensemble.

Nothing here runs on import. The completed two-fold early gate is checked
before any full-prefix/late-data access or outer-plan creation.
"""
import argparse
import ast
import copy
import gc
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import torch

import experiment_peer_dual_temporal as run
import experiment_peer_scale as scales
import experiment_peer_temporal_arch as previous
import experiment_target_peer_gru as graphs
import experiment_dynamic_peer_gru as peer
import experiment_y2_generalization as norm
import experiment_y2_gru_capacity as common
import experiment_y2_normalized_outer as outer
import experiment_mixed_market_routing as routing

OUT = run.OUT
SEEDS = (42, 43, 44)
TRAIN_DAYS = list(range(16))
LATE_DAYS = [16, 17, 18, 19]
ALLOWED_MODES = ('fixed_equal',)
ORIGINAL = previous.OUT / 'commit_peer_temporal'
GRAPH = graphs.OUT / 'outer_graph.npy'


def read_json(path):
    return json.loads(Path(path).read_text())


def validate_comparison(record, days):
    """Reject incomplete, nonfinite, or internally inconsistent saved metrics."""
    expected_days = {str(day) for day in days}
    if set(record['daily']) != expected_days:
        raise RuntimeError('Incomplete early daily metrics')
    for values in (record, *record['daily'].values()):
        triple = [values['base'], values['candidate'], values['delta']]
        if not np.isfinite(triple).all():
            raise RuntimeError('Nonfinite early gate metric')
        if abs(values['candidate'] - values['base'] - values['delta']) > 1e-10:
            raise RuntimeError('Inconsistent early metric difference')


def selected_mode(early):
    """Independently verify both complete folds before any full/late read."""
    mode = early.get('selected')
    if mode != 'fixed_equal' or early.get('eligibility', {}).get(mode, {}).get('passed') is not True:
        raise RuntimeError('Fixed equal temporal ensemble did not pass the early gate')
    if early.get('late_dates_evaluated') is not False:
        raise RuntimeError('Early evaluation must not contain late dates')
    positive = 0
    for fold, spec in common.FOLDS.items():
        record = early.get('folds', {}).get(fold)
        if record is None:
            raise RuntimeError('Incomplete early folds')
        try:
            validate_comparison(record, spec['validation'])
            if record['delta'] < .001:
                raise RuntimeError('Insufficient early gain over current TCN')
            positive += sum(value['delta'] > 0 for value in record['daily'].values())
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError('Malformed fixed-equal early comparison') from exc
    if positive < 6 or early['eligibility'][mode].get('daily_positive') != positive:
        raise RuntimeError('Insufficient positive daily gains')
    return mode


def paired_checkpoint_hashes(mode):
    if mode != 'fixed_equal':
        raise RuntimeError('Only the frozen equal mixture is permitted')
    return {fold: {architecture: run.verify_source(architecture, fold, 42)
                   for architecture in ('tcn_local', 'complex_lru')}
            for fold in common.FOLDS}


def early_gate():
    # Reject selected=null before even the runner's other artifact checks.
    early = read_json(OUT / 'pilot_results.json')
    mode = selected_mode(early)
    verified_early, plan = run.verify_gate()
    if early != verified_early:
        raise RuntimeError('Early decision changed during verification')
    if plan.get('mode') != 'fixed_equal' or plan.get('weights') != [.5, .5] or plan.get('folds') != common.FOLDS:
        raise RuntimeError('Frozen equal weights or early dates changed')
    baseline_pilot = read_json(previous.OUT / 'pilot_results.json')
    for fold, spec in common.FOLDS.items():
        base = baseline_pilot['folds'][fold]['tcn_local']
        record = early['folds'][fold]
        if abs(record['base'] - base['candidate']) > 1e-10:
            raise RuntimeError('Early baseline is not the current TCN')
        for day in spec['validation']:
            if abs(record['daily'][str(day)]['base'] - base['daily'][str(day)]['candidate']) > 1e-10:
                raise RuntimeError('Early daily baseline differs from current TCN')
    if plan['source_checkpoints'] != paired_checkpoint_hashes(mode):
        raise RuntimeError('Frozen early TCN/LRU source changed')
    for fold in common.FOLDS:
        controls = early.get('controls', {}).get(fold, {})
        if set(controls) != {'tcn_local', 'complex_lru'}:
            raise RuntimeError('Missing independent early model checks')
        means = []
        for architecture in ('tcn_local', 'complex_lru'):
            check = controls[architecture]
            expected_source = plan['source_checkpoints'][fold][architecture]
            if check.get('source') != expected_source or check.get('cpu_spread_rows', 0) < 50:
                raise RuntimeError('Early cached model source or sample verification differs')
            errors = [check.get('max_abs_heads', float('nan')), check.get('max_abs_final', float('nan'))]
            if not np.isfinite(errors).all() or min(errors) < 0 or errors[0] > 5e-5 or errors[1] > 1e-6:
                raise RuntimeError('Early CPU/cache verification did not pass')
            for ending, key in (('heads', 'head_prediction_sha256'), ('prediction', 'final_prediction_sha256')):
                path = previous.OUT / f'{architecture}_{fold}_{ending}.npy'
                actual = common.sha256(path)
                if check.get(key) != actual or plan['cache_sha256'].get(str(path)) != actual:
                    raise RuntimeError('Early cached prediction hash differs')
            means.append(expected_source['weight_mean'])
        if abs(means[0] - means[1]) > 1e-10:
            raise RuntimeError('Early TCN and LRU use different scale weights')
    baseline_report()
    return mode, early, plan


def baseline_report():
    report = read_json(previous.OUT / 'official_results.json')
    if report.get('promoted') is not True or report.get('mode') != 'tcn_local':
        raise RuntimeError('Current baseline must be promoted local TCN')
    if Path(report['commit']).resolve() != ORIGINAL.resolve() or len(report['hashes']) != 8:
        raise RuntimeError('Unexpected current TCN submission package')
    for name, expected in report['hashes'].items():
        if common.sha256(ORIGINAL / name) != expected:
            raise RuntimeError(f'Current best package changed: {name}')
    if run.verify_best() != report['hashes']:
        raise RuntimeError('Runner and outer disagree on current best package')
    return report


def freeze_plan(mode, early, pilot_plan):
    verified, _, _ = early_gate()
    if mode != verified or early != read_json(OUT / 'pilot_results.json'):
        raise RuntimeError('Early decision changed before outer-plan creation')
    old = baseline_report()
    sources = dict(pilot_plan['source_sha256'])
    for source in (__file__, run.__file__, previous.__file__, previous.lru.__file__, previous.tcn.__file__,
                   scales.__file__, graphs.__file__, peer.__file__, norm.__file__, common.__file__, outer.__file__, routing.__file__):
        sources[str(Path(source).resolve())] = common.sha256(source)
    plan = {
        'mode': mode, 'encoder_mode': 'tcn_local', 'dual_mode': 'fixed_equal', 'weights': [.5, .5],
        'train_days': TRAIN_DAYS, 'validation_days': LATE_DAYS, 'seeds': list(SEEDS), 'epochs': 2, 'window': 16,
        'training': {'architecture': 'complex_lru', 'lr': 1e-4, 'weight_decay': 1e-4, 'clip': 1.,
                     'row_batch': 1024, 'main_weights': [.375, .625], 'permutation': 'seed+epoch'},
        'parameters_per_lru': 203970,
        'tcn_sources': {str(seed): run.verify_source('tcn_local', 'outer', seed) for seed in SEEDS},
        'inference': 'Frozen TCN3seed mean and new LRU3seed mean, fixed .5/.5 before unchanged routing; both stored predictions already contain one causal S',
        'package': 'Preserve all154 original TCN package tensors, append102 LRU tensors; final total256',
        'baseline': str(ORIGINAL), 'baseline_official': old['pooled_official'], 'baseline_hashes': old['hashes'],
        'baseline_prediction_sha256': common.sha256(previous.OUT / 'outer_prediction.npy'),
        'baseline_raw_results_sha256': common.sha256(previous.OUT / 'outer_results.json'),
        'tcn_heads_sha256': common.sha256(previous.OUT / 'outer_joint_heads.npy'),
        'graph': str(GRAPH), 'graph_file_sha256': common.sha256(GRAPH),
        'training_fingerprint_plan': str(previous.OUT / 'plan.json'),
        'training_fingerprint_plan_sha256': common.sha256(previous.OUT / 'plan.json'),
        'pilot_plan_sha256': common.sha256(OUT / 'plan.json'),
        'pilot_results_sha256': common.sha256(OUT / 'pilot_results.json'),
        'pilot_checkpoint_sha256': paired_checkpoint_hashes(mode),
        'data_sha256': common.EXPECTED_SHA, 'source_sha256': sources,
        'late_dates_status': 'Repeatedly exposed retrospective validation, not blind',
        'promotion': 'Evaluate only fixed equal mixture; raw then official rounded score must strictly exceed current TCN; no late weight search',
    }
    path = OUT / 'outer_plan.json'
    if path.exists() and read_json(path) != plan:
        raise RuntimeError('Frozen outer plan mismatch')
    common.save_json(path, plan)
    return plan


def guarded_context(mode):
    selected, early, _ = early_gate()
    if mode != selected:
        raise RuntimeError('Requested mode is not the frozen equal ensemble')
    plan = read_json(OUT / 'outer_plan.json')
    if plan['mode'] != mode or plan['dual_mode'] != 'fixed_equal' or plan['weights'] != [.5, .5] or plan['encoder_mode'] != 'tcn_local':
        raise RuntimeError('Fixed architecture or weights changed')
    if plan['pilot_results_sha256'] != common.sha256(OUT / 'pilot_results.json') or plan['pilot_plan_sha256'] != common.sha256(OUT / 'plan.json'):
        raise RuntimeError('Frozen early decision changed')
    if plan['pilot_checkpoint_sha256'] != paired_checkpoint_hashes(mode):
        raise RuntimeError('Early source checkpoint changed')
    if plan['train_days'] != TRAIN_DAYS or plan['validation_days'] != LATE_DAYS or plan['seeds'] != list(SEEDS) or plan['epochs'] != 2:
        raise RuntimeError('Unexpected outer training specification')
    if plan['tcn_sources'] != {str(seed): run.verify_source('tcn_local', 'outer', seed) for seed in SEEDS}:
        raise RuntimeError('Preserved full-prefix TCN checkpoint changed')
    for name, expected in plan['source_sha256'].items():
        if common.sha256(name) != expected:
            raise RuntimeError(f'Outer source changed: {name}')
    if common.sha256(previous.OUT / 'plan.json') != plan['training_fingerprint_plan_sha256']:
        raise RuntimeError('Original LRU training fingerprint plan changed')
    if common.sha256(GRAPH) != plan['graph_file_sha256']:
        raise RuntimeError('Outer graph changed')
    old = baseline_report()
    if old['hashes'] != plan['baseline_hashes']:
        raise RuntimeError('Baseline package differs from frozen plan')
    for name, key in (('outer_prediction.npy', 'baseline_prediction_sha256'),
                      ('outer_results.json', 'baseline_raw_results_sha256'), ('outer_joint_heads.npy', 'tcn_heads_sha256')):
        if common.sha256(previous.OUT / name) != plan[key]:
            raise RuntimeError('Frozen baseline prediction artifact changed')
    return plan, early, old


def checkpoint_hashes(mode):
    if mode != 'fixed_equal':
        raise RuntimeError('Only fixed equal ensemble is supported')
    rows = 50 * sum(len(np.load(norm.CACHE / f'day_{day:02d}_x.npy', mmap_mode='r')) for day in TRAIN_DAYS)
    graph_sha = hashlib.sha256(np.asarray(np.load(GRAPH), dtype=np.float32).tobytes()).hexdigest()
    hashes = {}
    weight_mean = None
    for seed in SEEDS:
        path = OUT / f'complex_lru_outer_s{seed}.pth'
        meta = read_json(path.with_suffix('.json'))
        expected = {'mode': 'complex_lru', 'seed': seed, 'rows': rows, 'window': 16, 'epochs': 2, 'row_batch': 1024,
                    'plan_sha256': common.sha256(previous.OUT / 'plan.json'), 'graph_sha256': graph_sha}
        if meta.get('fingerprint') != expected or meta.get('parameters') != 203970:
            raise RuntimeError('Full LRU fingerprint differs from the frozen original training recipe')
        records = meta.get('records', [])
        if len(records) != 2 or any(r.get('rows') != rows or r.get('batches') != (rows + 1023) // 1024 or
                                  not np.isfinite(r.get('weighted_loss', float('nan'))) for r in records):
            raise RuntimeError('Incomplete LRU full-prefix training')
        value = meta.get('weight_mean', float('nan'))
        if not np.isfinite(value) or value <= 0:
            raise RuntimeError('Invalid LRU training normalization')
        if weight_mean is not None and abs(value - weight_mean) > 1e-10:
            raise RuntimeError('LRU seeds use different scale weights')
        weight_mean = value
        if abs(value - run.verify_source('tcn_local', 'outer', seed)['weight_mean']) > 1e-10:
            raise RuntimeError('TCN and LRU training weights differ')
        verified = run.verify_source('complex_lru', 'outer', seed)
        if Path(verified['checkpoint']).resolve() != path.resolve() or verified['checkpoint_sha256'] != common.sha256(path) or meta['sha256'] != verified['checkpoint_sha256']:
            raise RuntimeError('Full LRU checkpoint SHA mismatch')
        hashes[str(seed)] = meta['sha256']
    return hashes


def train_evaluate(src, mode):
    plan, early, _ = guarded_context(mode)
    plan_sha = common.sha256(OUT / 'outer_plan.json')
    cached = OUT / 'outer_results.json'
    if cached.exists():
        result = read_json(cached)
        if result['mode'] != mode or result['outer_plan_sha256'] != plan_sha or result['checkpoint_sha256'] != checkpoint_hashes(mode):
            raise RuntimeError('Cached outer result mismatch')
        for name, expected in result['prediction_sha256'].items():
            if common.sha256(OUT / name) != expected:
                raise RuntimeError('Cached outer predictions changed')
        return result
    ref = scales.reference()
    graph = np.asarray(np.load(GRAPH), dtype=np.float32)
    raw, y, tid = norm.load_days(TRAIN_DAYS, normalized=True)
    raw = raw.astype(np.float32)
    X = peer.PeerArray(raw, graphs.context(raw, graph))
    scale = scales.load_scale(TRAIN_DAYS, ref)
    if len(X) != len(y) or len(tid) != len(X) or len(scale) != len(X):
        raise RuntimeError('Full-prefix training arrays are misaligned')
    print(json.dumps({'stage': 'outer_training_loaded', 'mode': mode, 'rows': len(X), 'seeds': list(SEEDS)}), flush=True)
    for seed in SEEDS:
        model = previous.train(src, X, y, tid, scale, graph, 'complex_lru', seed, OUT / f'complex_lru_outer_s{seed}.pth')
        del model
        gc.collect()
        torch.cuda.empty_cache()
    hashes = checkpoint_hashes(mode)
    expected_weight = float(np.mean(np.square(scale.astype(np.float64)) ** -1))
    for seed in SEEDS:
        meta = read_json(OUT / f'complex_lru_outer_s{seed}.json')
        if abs(meta['weight_mean'] - expected_weight) > 1e-10:
            raise RuntimeError('Outer checkpoint weight normalization differs from full-prefix data')
    del raw, X, y, tid, scale
    gc.collect()

    # This is the first read of late raw features/labels, after the early gate.
    raw, y, tid = src.load_block(LATE_DAYS)
    if len(raw) % 50 or len(raw) != len(tid) or len(y) != len(raw):
        raise RuntimeError('Late frame alignment mismatch')
    starts = np.flatnonzero(tid[::50] == 0) * 50
    if len(starts) != 4 or starts[0] != 0:
        raise RuntimeError('Expected exactly four independently reset late days')
    boundaries = np.concatenate((starts, [len(raw)]))
    day_rows = np.diff(boundaries)
    mu, sd = norm.global_stats()
    parts = [outer.transform(raw[a:b], mu, sd) for a, b in zip(boundaries[:-1], boundaries[1:])]
    del raw
    raw = np.concatenate(parts).astype(np.float32)
    del parts
    X = peer.PeerArray(raw, graphs.context(raw, graph))
    scale = scales.load_scale(LATE_DAYS, ref)
    if len(scale) != len(X):
        raise RuntimeError('Late scale alignment mismatch')
    predictions = []
    for seed in SEEDS:
        model = previous.build('complex_lru').to('cuda:0')
        model.load_state_dict(torch.load(OUT / f'complex_lru_outer_s{seed}.pth', map_location='cuda:0', weights_only=True), strict=True)
        # Already multiplied by the causal scale inside predict: do not rescale.
        predictions.append(run.predict(src, model, X, tid, scale))
        del model
        gc.collect()
        torch.cuda.empty_cache()
        print(json.dumps({'stage': 'outer_prediction', 'mode': mode, 'seed': seed}), flush=True)
    lru_heads = np.stack(predictions).mean(0)
    tcn_heads = np.load(previous.OUT / 'outer_joint_heads.npy')
    if tcn_heads.shape != lru_heads.shape or not np.isfinite(tcn_heads).all():
        raise RuntimeError('Frozen TCN prediction shape or values invalid')
    q = .5 * (tcn_heads + lru_heads)
    np.save(OUT / 'outer_lru_heads.npy', lru_heads)
    if q.shape != (len(y), 2) or not np.isfinite(q).all():
        raise RuntimeError('Invalid ensemble predictions')
    np.save(OUT / 'outer_joint_heads.npy', q)
    del X, raw, tid, predictions
    gc.collect()
    with np.load(outer.OUT / 'validation_components.npz') as z:
        if not np.array_equal(y, z['target']) or not np.array_equal(day_rows, z['day_rows']):
            raise RuntimeError('Archived target/day alignment mismatch')
    base = np.load(previous.OUT / 'outer_prediction.npy')
    if base.shape != y.shape:
        raise RuntimeError('Baseline prediction shape mismatch')
    route_error = float(np.max(np.abs(routing.routed(base[:, 1], tcn_heads[:, 1]) - base[:, 2])))
    if not np.isfinite(route_error) or route_error > 1e-6:
        raise RuntimeError('Frozen TCN heads do not reproduce the current baseline')
    candidate = base.copy()
    candidate[:, 2] = routing.routed(base[:, 1], q[:, 1])
    np.testing.assert_array_equal(candidate[:, :2], base[:, :2])
    np.save(OUT / 'outer_prediction.npy', candidate)
    result = {
        'mode': mode, 'outer_plan_sha256': plan_sha, 'checkpoint_sha256': hashes,
        'baseline': outer.metrics(base, y), 'candidate': outer.metrics(candidate, y),
        'direct_head_corr_y2': norm.corr(q[:, 1], y[:, 2]), 'lru_direct_head_corr_y2': norm.corr(lru_heads[:, 1], y[:, 2]),
        'frozen_tcn_route_max_abs_error': route_error, 'daily': {},
        'diagnostic': scales.evaluate_prediction(candidate[:, 2], base[:, 2], y, LATE_DAYS, scale),
        'early_gate': early, 'late_dates_status': plan['late_dates_status'], 'day_rows': day_rows.tolist(),
    }
    expected = read_json(previous.OUT / 'outer_results.json')['candidate']
    for key in ('corr', 'score'):
        np.testing.assert_allclose(result['baseline'][key], expected[key], rtol=0, atol=1e-10)
    offset = 0
    for day, n in zip(LATE_DAYS, day_rows):
        sl = slice(offset, offset + int(n))
        b, c = outer.metrics(base[sl], y[sl]), outer.metrics(candidate[sl], y[sl])
        result['daily'][str(day)] = {'baseline': b, 'candidate': c, 'score_delta': c['score'] - b['score']}
        offset += int(n)
    result['prediction_sha256'] = {name: common.sha256(OUT / name) for name in ('outer_prediction.npy', 'outer_joint_heads.npy', 'outer_lru_heads.npy')}
    baseline_report()
    common.save_json(cached, result)
    print(json.dumps({'stage': 'outer_complete', 'mode': mode, 'baseline': result['baseline'], 'candidate': result['candidate']}), flush=True)
    return result


def append_lru_members(original_state, trained_states, graph, mode):
    """Preserve all154 original tensors and append three complete LRU cores."""
    a = torch.as_tensor(graph, dtype=torch.float32, device='cpu')
    if mode != 'fixed_equal' or a.shape != (50, 50) or not torch.equal(original_state['peer_graph'], a):
        raise RuntimeError('Fixed ensemble or graph mismatch')
    if len(trained_states) != 3 or len(original_state) != 154:
        raise RuntimeError('Unexpected preserved TCN schema or seed count')
    tcn_keys = previous.build('tcn_local').state_dict()
    expected_tcn = {f'members128.{i}.{key}' for i in range(3) for key in tcn_keys}
    if len(tcn_keys) != 38 or {key for key in original_state if key.startswith('members128.')} != expected_tcn:
        raise RuntimeError('Current TCN members are not the expected114 tensors')
    if any(key.startswith('members_lru.') for key in original_state):
        raise RuntimeError('Baseline already contains an LRU branch')
    state = dict(original_state)
    inserted = set()
    for i, trained in enumerate(trained_states):
        model = previous.build('complex_lru')
        expected = model.state_dict()
        if len(expected) != 34 or set(trained) != set(expected):
            raise RuntimeError('Unexpected LRU state keys')
        for key, value in trained.items():
            if not isinstance(value, torch.Tensor) or value.shape != expected[key].shape or value.dtype != expected[key].dtype or not torch.isfinite(value).all():
                raise RuntimeError(f'Invalid LRU tensor: {key}')
        model.load_state_dict(trained, strict=True)
        for key, value in trained.items():
            destination = f'members_lru.{i}.{key}'
            if destination in state:
                raise RuntimeError('Duplicate LRU deployment key')
            state[destination] = value
            inserted.add(destination)
    if len(inserted) != 102 or len(state) != 256 or set(state) != set(original_state) | inserted:
        raise RuntimeError('Unexpected parallel ensemble state schema')
    for key, value in original_state.items():
        if not torch.equal(state[key], value):
            raise RuntimeError(f'Original TCN package tensor changed: {key}')
    return state, sorted(inserted)


def export_network_source(mode):
    if mode != 'fixed_equal':
        raise RuntimeError('Only fixed equal ensemble may be exported')
    original = (ORIGINAL / 'network.py').read_text()
    original_tree = ast.parse(original)
    required = ('ComplexDiagonalLRU', 'LRUBlock', 'PeerLRU')
    helper_tree = ast.parse(Path(previous.lru.__file__).read_text())
    nodes = [copy.deepcopy(n) for n in helper_tree.body if isinstance(n, ast.ClassDef) and n.name in required]
    if tuple(n.name for n in nodes) != required:
        raise RuntimeError('Unexpected LRU helper layout')
    existing = {n.name for n in original_tree.body if isinstance(n, ast.ClassDef)}
    if 'TemporalScalePeerTarget' not in existing or existing.intersection(required) or 'DualTemporalScalePeerTarget' in existing:
        raise RuntimeError('Missing TCN dependency or duplicate LRU class')
    replacements = []
    class RemoveReference(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'reference':
                if ast.unparse(node.value) != 'peer.PeerGRU()':
                    raise RuntimeError('Unexpected training reference')
                replacements.append('reference')
                return None
            if isinstance(node.value, ast.Attribute) and isinstance(node.value.value, ast.Name) and node.value.value.id == 'reference':
                head = node.value.attr
                if head not in ('head1', 'head2') or ast.unparse(node.targets[0]) != 'self.' + head:
                    raise RuntimeError('Unexpected training reference head')
                replacements.append(head)
                node.value = ast.parse('nn.Linear(128, 1)', mode='eval').body
            return self.generic_visit(node)
    nodes = [RemoveReference().visit(n) for n in nodes]
    if sorted(replacements) != ['head1', 'head2', 'reference']:
        raise RuntimeError('LRU training reference not fully removed')
    for node in nodes:
        if any(isinstance(n, ast.Name) and n.id in ('peer', 'reference', 'run', 'previous') for n in ast.walk(node)):
            raise RuntimeError('LRU deployment still depends on training project')
    parent = next(n for n in original_tree.body if isinstance(n, ast.ClassDef) and n.name == 'TemporalScalePeerTarget')
    predict = copy.deepcopy(next(n for n in parent.body if isinstance(n, ast.FunctionDef) and n.name == 'predict'))
    expected = ast.parse('torch.stack([m(history)[:, 1:2] for m in self.members128]).mean(0)', mode='eval').body
    mixture = ast.parse('0.5 * (torch.stack([m(history)[:, 1:2] for m in self.members128]).mean(0) + torch.stack([m(history)[:, 1:2] for m in self.members_lru]).mean(0))', mode='eval').body
    changed = []
    class EqualMixture(ast.NodeTransformer):
        def visit_Assign(self, node):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) and node.targets[0].id == 'p128' and ast.dump(node.value) == ast.dump(expected):
                node.value = copy.deepcopy(mixture)
                changed.append(True)
            return self.generic_visit(node)
    predict = EqualMixture().visit(predict)
    if len(changed) != 1:
        raise RuntimeError('Expected exactly one raw-head mixture change')
    initializer = ast.parse('def __init__(self, config):\n    if config.get("encoder_mode") != "tcn_local" or config.get("dual_mode") != "fixed_equal" or config.get("dual_weights") != [0.5, 0.5]:\n        raise ValueError("Unexpected frozen temporal mixture")\n    super().__init__(config)\n    self.members_lru = nn.ModuleList([PeerLRU() for _ in range(3)])\n').body[0]
    deployment = ast.ClassDef(name='DualTemporalScalePeerTarget', bases=[ast.Name(id='TemporalScalePeerTarget', ctx=ast.Load())],
                              keywords=[], body=[initializer, predict], decorator_list=[])
    exported = ast.fix_missing_locations(ast.Module(body=nodes + [deployment], type_ignores=[]))
    source = original + '\nimport math\n' + ast.unparse(exported) + '\n'
    ast.parse(source)
    return source


def package(src, result):
    mode = result['mode']
    _, early, old = guarded_context(mode)
    if result.get('outer_plan_sha256') != common.sha256(OUT / 'outer_plan.json'):
        raise RuntimeError('Outer result belongs to another frozen plan')
    if result != read_json(OUT / 'outer_results.json') or result['checkpoint_sha256'] != checkpoint_hashes(mode):
        raise RuntimeError('Outer result/checkpoint changed before packaging')
    for name, expected in result['prediction_sha256'].items():
        if common.sha256(OUT / name) != expected:
            raise RuntimeError('Outer prediction artifact changed before packaging')
    if result['candidate']['score'] <= result['baseline']['score']:
        report = {'promoted': False, 'reason': 'Raw retrospective score did not exceed the previous best',
                  'mode': mode, 'pooled_official': None, 'baseline_official': old['pooled_official'],
                  'candidate_vector': result['candidate'], 'early_gate': early,
                  'baseline_hashes_unchanged': old['hashes'], 'note': result['late_dates_status']}
        common.save_json(OUT / 'official_results.json', report)
        print(json.dumps({'stage': 'not_promoted', 'reason': report['reason']}), flush=True)
        return report
    graph = np.asarray(np.load(GRAPH), dtype=np.float32)
    trained = []
    for seed in SEEDS:
        weights = torch.load(OUT / f'complex_lru_outer_s{seed}.pth', map_location='cpu', weights_only=True)
        model = previous.build('complex_lru')
        model.load_state_dict(weights, strict=True)
        trained.append(weights)
    original_state = torch.load(ORIGINAL / 'model/Model_2.pth', map_location='cpu', weights_only=True)
    state, inserted = append_lru_members(original_state, trained, graph, mode)
    commit = OUT / 'commit_peer_dual_temporal'
    # Start from the eight declared package files; only target2 code/config/weights may change.
    for name in old['hashes']:
        destination = commit / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ORIGINAL / name, destination)
    (commit / 'network.py').write_text(export_network_source(mode))
    config = read_json(ORIGINAL / 'conf/Model_2.json')
    config.update(model='DualTemporalScalePeerTarget', dual_mode='fixed_equal', dual_weights=[.5, .5])
    common.save_json(commit / 'conf/Model_2.json', config)
    torch.save(state, commit / 'model/Model_2.pth')
    allowed_changes = {'network.py', 'conf/Model_2.json', 'model/Model_2.pth'}
    for name in old['hashes']:
        if name not in allowed_changes and common.sha256(commit / name) != old['hashes'][name]:
            raise RuntimeError(f'Unchanged package file differs: {name}')
    from script.evaluate import Evaluator
    evaluator = Evaluator(str(common.DATA), str(commit), 'cuda:0')
    expected = np.load(OUT / 'outer_prediction.npy')
    base = np.load(previous.OUT / 'outer_prediction.npy')
    np.testing.assert_array_equal(expected[:, :2], base[:, :2])
    maximum = 0.
    offset = 0
    for day, rows in zip(LATE_DAYS, result['day_rows']):
        if day not in (16, 17):
            break
        raw, _, _ = src.load_block([day])
        if len(raw) != rows or len(raw) < 128 * 50:
            raise RuntimeError('Stream verification day shape mismatch')
        for model in evaluator.models:
            model.reset()
        observed = []
        with torch.no_grad():
            for frame in raw[:128 * 50].reshape(-1, 50, 100):
                inp = torch.from_numpy(frame).to('cuda:0')
                observed.append(torch.cat([model.predict(inp.clone()) for model in evaluator.models], dim=1).cpu().numpy())
        observed = np.concatenate(observed)
        error = float(np.abs(observed - expected[offset:offset + len(observed)]).max())
        if not np.isfinite(error) or error > 5e-4:
            raise RuntimeError(f'Stream/batch mismatch on day {day}: {error}')
        maximum = max(maximum, error)
        offset += int(rows)
        del raw, observed
        gc.collect()
    print(json.dumps({'stage': 'streaming_preflight', 'days': [16, 17], 'ticks_per_day': 128, 'max_abs_error': maximum}), flush=True)
    score = evaluator.evaluate(date=LATE_DAYS)
    keys = ('corr0', 'corr1', 'corr2', 'score')
    if any(not np.isfinite(score[key]) or abs(score[key] - result['candidate']['rounded'][key]) > .00011 for key in keys):
        raise RuntimeError('Official/vector score mismatch')
    if any(score[key] != old['pooled_official'][key] for key in ('corr0', 'corr1')):
        raise RuntimeError('Unchanged target score changed')
    baseline_report()
    promoted = score['score'] > old['pooled_official']['score']
    report = {
        'promoted': promoted, 'mode': mode, 'pooled_official': score,
        'baseline_official': old['pooled_official'], 'delta_score': round(score['score'] - old['pooled_official']['score'], 4),
        'commit': str(commit), 'encoder_mode': 'tcn_local', 'dual_mode': mode, 'streaming_max_abs_error': maximum,
        'preserved_tensor_count': len(original_state), 'inserted_member_keys': inserted,
        'early_gate': early, 'daily': result['daily'], 'note': result['late_dates_status'],
        'outer_plan_sha256': common.sha256(OUT / 'outer_plan.json'),
        'checkpoint_sha256': result['checkpoint_sha256'], 'baseline_hashes_unchanged': old['hashes'],
        'hashes': {name: common.sha256(commit / name) for name in old['hashes']},
    }
    if not promoted:
        report['reason'] = 'Official rounded score did not exceed the previous best; package is an unpromoted candidate'
    common.save_json(OUT / 'official_results.json', report)
    print(json.dumps({'stage': 'complete' if promoted else 'not_promoted', 'mode': mode, 'pooled_official': score,
                      'delta_score': report['delta_score'], 'commit': str(commit)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check-only', action='store_true', help='Verify only completed early artifacts; do not create an outer plan or access full/late data')
    args = parser.parse_args()
    mode, early, pilot_plan = early_gate()
    print(json.dumps({'stage': 'early_gate_verified', 'mode': mode}), flush=True)
    if args.check_only:
        return
    freeze_plan(mode, early, pilot_plan)
    if common.sha256(common.DATA) != common.EXPECTED_SHA:
        raise RuntimeError('Data hash mismatch')
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(4)
    src, _ = outer.prepare()
    result = train_evaluate(src, mode)
    package(src, result)


if __name__ == '__main__':
    main()
