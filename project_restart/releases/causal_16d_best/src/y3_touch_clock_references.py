"""Replay the touch-flow addition from verified parent grids and rebuild its training folds."""
import json
import sys

import numpy as np
import torch

import experiment_y3_touch_flow_clocks_v1 as incumbent
from y3_expanding_path_references import read_verified
from y3_same_deadline_references import rebuild

previous = incumbent.history
trial, study, stack = previous.trial, previous.study, previous.stack
CACHE = incumbent.OUT / 'historical_references'
FULL = incumbent.OUT / 'full_chain_verification'
digest = previous.previous.previous.digest


def references(day, raw):
    return read_verified(CACHE, day, raw)


def read(day, raw):
    return read_stage(FULL, day, raw)


def read_stage(directory, day, raw):
    with np.load(directory / f'day_{day}.npz') as archive:
        q, h = archive['full'], archive['head']
        for key, data in [('raw', raw), ('full', q), ('head', h)]:
            assert str(archive[key + '_hash']) == digest(data)
    return q, h, len(h)


def verify():
    with np.load(incumbent.OUT/'model.npz') as loaded:
        state = dict(loaded)
    weight = state['w'][:, int(state['pick'])]
    def predict(raw, p):
        z = incumbent.signals(raw, np.arange(len(raw)), state, state['heads'])
        x = np.concatenate((trial.clock.center(p), z), axis=2)
        update = np.einsum('tsi,i->ts', x, weight, optimize=False)
        q = p.copy()
        q[:, :, 2] += update
        return q
    verify_stage(incumbent, previous.read, predict)


def verify_stage(module, loader, predict):
    directory = module.OUT/'full_chain_verification'
    directory.mkdir(exist_ok=True)
    trace = stack.verifier.trace
    expected = json.loads((module.OUT/'reload_verification.json').read_text())
    standard = module.correction()
    rows, daily = [], {}
    for d in range(16, 20):
        raw = trace.replay.original.features(trace.replay.original.DATA, d)
        p, h, n = loader(d, raw)
        q, head = predict(raw, p), predict(raw[:n], h)
        error = float(np.max(np.abs(head-q[:n])))
        assert error < 1e-4 and np.isfinite(q).all()
        # Recreate the original accumulation order before caching descendants' references.
        p1 = p.copy()
        p1[:, :, 2] += standard(raw, p)
        h1 = h.copy()
        h1[:, :, 2] += standard(raw[:n], h)
        np.testing.assert_allclose(p1, q, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(h1, head, rtol=1e-6, atol=1e-6)
        pair = (p.reshape(-1, 3), p1.reshape(-1, 3), trace.replay.labels(d))
        rows.append(pair)
        s = trace.older.measure([pair])
        assert abs(s['score']-expected['daily'][str(d+1)]['score']) < 1e-10
        daily[str(d+1)] = dict(score=s['score'], gain=s['gain'], prefix_error=error)
        np.savez(directory/f'day_{d}.npz', full=p1, head=h1, raw_hash=digest(raw),
                 full_hash=digest(p1), head_hash=digest(h1))
    pooled = trace.older.measure(rows)
    assert pooled['rows'] == 936250 and abs(pooled['score']-expected['pooled']['score']) < 1e-10
    (directory/'composed_replay.json').write_text(json.dumps(dict(pooled=pooled, daily=daily), indent=2))
    print('verified', pooled['score'], flush=True)


def replay_with():
    return stack.verifier, read


def outer(module):
    stack.verifier.outer(module, reference=incumbent, load_reference=read)


if __name__ == '__main__':
    torch.set_num_threads(2)
    if sys.argv[1:] == ['verify']:
        verify()
    else:
        rebuild(incumbent, previous.references, CACHE)
