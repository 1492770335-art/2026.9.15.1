"""Check causality and independently reconstruct the frozen, selected single-side print feature."""
import json
from unittest.mock import patch

import numpy as np
import torch

import experiment_y3_inferred_print_impacts_v1 as model


def run():
    full = model.trial.clock.book.rawday(9)
    pred = model.history.references(9, full)[0, :841]
    raw = full[:len(pred)]
    with np.load(model.OUT/'model.npz') as f:
        state = dict(f)
    assert int(state['pick'])==3
    update = model.correction()
    cuts = (1, 49, 119, 120, 121, 239, 240, 417, 710)
    with patch.object(model, 'parameters', side_effect=AssertionError('accessed training parameters')), \
         patch.object(model.trial.clock.book, 'rawday', side_effect=AssertionError('accessed other dates')), \
         patch.object(model.trial.clock, 'labels', side_effect=AssertionError('accessed targets')):
        delta = update(raw, pred)
        errors = [float(np.max(abs(delta[:n]-update(raw[:n], pred[:n])))) for n in cuts]
        assert max(errors)<1e-6
    mid, width, offset, valid = model.book.quote.infer(raw, float(state['k']), state['coefficients'])
    x = model.design_with(raw, pred, np.arange(len(raw)), state)
    expected = np.zeros((len(raw), 50, 2))
    for stock in range(50):
        for tick in range(1, len(raw)):
            if not (valid[tick-1, stock] and valid[tick, stock]):
                continue
            a, b = raw[tick, stock, 54:56]>state['floors'][1:]
            if a==b:
                continue
            side = 1 if a else -1
            if mid[tick, stock]+offset[tick, stock]!=mid[tick-1, stock]+side*width[tick-1, stock]:
                continue
            volume = max(0., float(raw[tick, stock, 8])-state['floors'][0])
            expected[tick, stock] = side, side*min(6., np.log1p(volume/state['volume_scale'][stock]))
    checked = np.arange(0, len(raw), 7)
    rows = []
    for tick in checked:
        frame = np.concatenate([expected[max(0, tick+1-h):tick+1].sum(0)/np.sqrt(h) for h in (5, 20, 60, 180)], axis=1)
        rows.append(frame-frame.mean(0))
    error = np.max(abs(np.asarray(rows)-x[checked, :, 27:35]))
    assert error<2e-6
    r = dict(cuts=cuts, prefix_errors=errors, feature_error=float(error), rows=len(checked)*50,
             accepted=int(np.count_nonzero(expected[:, :, 0])))
    (model.OUT/'frozen_inference_checks.json').write_text(json.dumps(r, indent=2))
    print(r, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    run()
