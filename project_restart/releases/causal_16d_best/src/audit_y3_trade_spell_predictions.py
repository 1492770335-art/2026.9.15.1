"""Test on later training dates whether run length helps predict future same-side event counts."""
import json

import numpy as np

import experiment_y3_trade_spell_survival_v1 as model


def run():
    rows = []
    for d in range(8, 16):
        params = model.parameters(d-d%2)
        raw = model.trial.clock.book.rawday(d)
        action, side, valid, broad, cell, length = model.state(raw, params)
        t = np.arange(120+d%4, len(raw)-60, 4)
        at = np.arange(len(raw))[:, None]
        direction = side[t]
        change = np.full(direction.shape, len(raw), dtype=int)
        for sign in (1, -1):
            back = np.minimum.accumulate(np.where(action==-sign, at, len(raw))[::-1], axis=0)[::-1]
            change = np.where(direction==sign, back[t+1], change)
        cs = np.vstack((np.zeros(50), action.cumsum(0)))
        vals = []
        for j, (lo, hi) in enumerate(model.HORIZONS):
            end = np.minimum(change, t[:, None]+hi+1)
            begin = np.minimum(change, t[:, None]+lo+1)
            actual = (cs[end, np.arange(50)]-cs[begin, np.arange(50)])*direction
            good = valid[t]
            pred = params['cells'][cell[t], j]
            base = params['broad'][broad[t], j]
            vals.append([int(good.sum()), float(np.sum((actual[good]-base[good])**2)),
                         float(np.sum((actual[good]-pred[good])**2))])
        rows.append(vals)
    z = np.sum(rows, axis=0)
    report = dict(days=list(range(9, 17)), forecast_intervals=model.HORIZONS, rows=z[:, 0].astype(int).tolist(),
                  squared_error_ratio=(z[:, 2]/z[:, 1]).tolist())
    (model.OUT/'chronological_flow_audit.json').write_text(json.dumps(report, indent=2))
    print(report)


if __name__ == '__main__':
    run()
