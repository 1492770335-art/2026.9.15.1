"""Check event-token inference against a prefix and a per-stock online history."""
import json
from unittest.mock import patch

import numpy as np
import torch

import experiment_y3_quote_event_tokens_v1 as events


def run():
    raw = events.trial.clock.book.rawday(9)
    p = events.history.references(9, raw)[0, :841]
    raw = raw[:841]
    with np.load(events.OUT/'model.npz') as f:
        state = dict(f)
    update = events.correction()
    cuts = (1, 119, 120, 121, 239, 240, 241, 417, 710)
    with patch.object(events.model, 'parameters', side_effect=AssertionError('attempted calibration')), \
         patch.object(events.trial.clock.book, 'rawday', side_effect=AssertionError('requested train data')), \
         patch.object(events.trial.clock, 'labels', side_effect=AssertionError('requested labels')):
        full = update(raw, p)
        differences = [float(np.max(abs(update(raw[:n], p[:n])-full[:n]))) for n in cuts]
        assert max(differences) < 1e-6
        features = events.design_with(raw, p, np.arange(len(raw)), state)[:, :, 3:].reshape(len(raw), 50, 2, 4, 7)
        mid, width, _, valid = events.model.quote.infer(raw, float(state['k']), state['coefficients'])
        checked = 0
        for side, level in enumerate((mid-width, mid+width)):
            for stock in (0, 14, 37, 49):
                last, memory = {}, []
                x = np.arcsinh(raw[:, stock, state['sides'][side]].astype('float32'))
                for t in range(len(raw)):
                    if valid[t, stock]:
                        price = int(level[t, stock])
                        prior = last.get(price)
                        if prior is not None and 1 < t-prior <= 240:
                            memory.insert(0, (t, prior))
                            memory = memory[:4]
                        last[price] = t
                    expected = np.zeros((4, 7), dtype='float32')
                    for j, (arrival, prev) in enumerate(memory):
                        if t-arrival <= 240:
                            expected[j] = np.r_[x[arrival]-x[prev], np.log1p(arrival-prev), np.log1p(t-arrival), 1]
                    np.testing.assert_allclose(features[t, stock, side], expected, rtol=0, atol=1e-6)
                    checked += 1
    result = dict(prefix_lengths=cuts, max_errors=differences, checked_event_arrays=checked)
    (events.OUT/'frozen_inference_checks.json').write_text(json.dumps(result, indent=2))
    print(result, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(2)
    run()
