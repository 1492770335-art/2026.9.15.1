"""Drop each price-error channel at fixed fold weights on training-period out-of-fold forecasts."""
import json
import numpy as np
import torch

import experiment_y3_unfulfilled_price_moves_v1 as model

def main():
    history = model.warm_history()
    chosen = json.loads((model.OUT / 'results.json').read_text())['selected']
    if model.ARMS[chosen] != 'three types':
        raise ValueError('This audit is defined for the three-type readout')
    drops = [None, 2, 3, 4]
    names = ('all', 'without_missed', 'without_reversed', 'without_overshoot')
    pooled = [[], []]
    for day in range(8, 16):
        raw = model.trial.clock.book.rawday(day)
        truth = model.trial.clock.labels(day).reshape(-1, 50, 3)
        state = model.historical_state(raw, day)
        if day % 2 == 0:
            weights = model.fit(history)
        records = []
        for i, p in enumerate(model.history.references(day, raw)):
            changes = []
            for j in drops:
                w = weights['w'][:, chosen:chosen+1].copy()
                if j is not None:
                    w[3+2*j:5+2*j] = 0
                changes.append(model.changes_with_reference(state, {'w': w}, p)[0])
            moments = model.trial.previous.statistics(p, truth, changes)
            pooled[i].append([m.sum(0) for m in moments])
            records.append(model.record_with_state(state, p, truth, day))
        history.append([np.mean(part, axis=0) for part in zip(*records)])
    reports = []
    for ref in pooled:
        total = np.sum(ref, axis=0)
        reports.append({name: dict(zip(names, model.trial.scoring.audit.pooled(g).tolist()))
                        for name, g in zip(('y3_gain','last40_gain'), total)})
    out = dict(days=list(range(9,17)), references=reports)
    (model.OUT / 'training_channel_audit.json').write_text(json.dumps(out,indent=2))
    print(out, flush=True)

if __name__ == '__main__':
    torch.set_num_threads(2)
    main()
