"""Compare vectorized spell states to a forward loop and prohibit training access during inference."""
import json
from unittest.mock import patch

import numpy as np
import torch

import experiment_y3_trade_spell_survival_v1 as model


def run():
    raw = model.trial.clock.book.rawday(9)
    p = model.history.references(9, raw)[0, :841]
    raw = raw[:len(p)]
    with np.load(model.OUT/'model.npz') as f:
        state = dict(f)
    assert int(state['pick'])==4
    update = model.correction()
    cuts = (1, 5, 119, 120, 121, 239, 240, 417, 710)
    with patch.object(model, 'parameters', side_effect=AssertionError('attempted to fit forward counts')), \
         patch.object(model.trial.clock.book, 'rawday', side_effect=AssertionError('requested train data')), \
         patch.object(model.trial.clock, 'labels', side_effect=AssertionError('requested labels')):
        full = update(raw, p)
        errors = [float(np.max(abs(update(raw[:n], p[:n])-full[:n]))) for n in cuts]
        assert max(errors)<1e-6
        action, sign, alive, broad, cell, length = model.state(raw, state)
        for s in (0, 13, 32, 49):
            count, start, prev, direction = 0, 0, -1, 0
            for t, a in enumerate(action[:, s]):
                if a:
                    if a!=direction:
                        count, start, direction = 0, t, a
                    count += 1
                    prev = t
                assert length[t, s]==count and sign[t, s]==direction
                assert alive[t, s]==(prev>=0 and t-prev<=80)
                volume = np.count_nonzero(action[max(0,t-59):t+1, s])
                coarse = 4*np.searchsorted([1,3,8,20], t-prev, side='right')+np.searchsorted([4,11,22], volume, side='right')
                pace = max(0, prev-start)/max(1, count-1)
                code = (coarse*5+np.searchsorted([2,3,5,9], count, side='right'))*3+np.searchsorted([2,7], pace, side='right')
                assert broad[t, s]==coarse and cell[t, s]==code
    report = dict(cuts=cuts, prefix_errors=errors, checked_states=4*len(raw))
    (model.OUT/'frozen_inference_checks.json').write_text(json.dumps(report, indent=2))
    print(report, flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
