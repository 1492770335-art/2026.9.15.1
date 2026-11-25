"""Check the inferred price offset against independently observed trade activity and side quantities."""
import json

import numpy as np

import y3_quote_token_references as history

book, trial = history.incumbent.model, history.trial
OUT = history.incumbent.OUT


def observations(raw, state, floor):
    mid, width, offset, ok = book.quote.infer(raw, float(state['k']), state['coefficients'])
    buy, sell = raw[:, :, 54]>floor[0], raw[:, :, 55]>floor[1]
    previous = np.r_[0, np.arange(len(raw)-1)]
    valid = ok & ok[previous]
    valid[0] = False
    result = {}
    for j, loc in enumerate((mid+offset, mid-offset, mid)):
        single = buy^sell
        face = np.where(buy, mid+width, mid-width)
        priorface = np.where(buy, (mid+width)[previous], (mid-width)[previous])
        against = 2*mid-face
        back = np.where(buy, (mid-width)[previous], (mid+width)[previous])
        tests = (('stationary_inactive', valid & ~(buy|sell), loc==loc[previous]),
                 ('stationary_inactive_moving_mid', valid & ~(buy|sell) & (mid!=mid[previous]), loc==loc[previous]),
                 ('stationary_active_moving_mid', valid & (buy|sell) & (mid!=mid[previous]), loc==loc[previous]),
                 ('same_side_now', valid & single, loc==face),
                 ('opposite_side_now', valid & single, loc==against),
                 ('same_side_before', valid & single, loc==priorface),
                 ('opposite_side_before', valid & single, loc==back))
        result[j] = [(name, int(np.sum(mask & hit)), int(mask.sum())) for name, mask, hit in tests]
    return result


def run():
    state = book.parameters()
    floor = trial.clock.book.rawday(0)[:, :, [54, 55]].min((0, 1))
    rows = [observations(trial.clock.book.rawday(d), state, floor) for d in range(16)]
    report = {}
    for name, split in (('training_discovery', rows[:4]), ('later_training_dates', rows[8:])):
        modes = {}
        for j, loc in enumerate(('mid_plus', 'mid_minus', 'mid_only')):
            counts = np.sum([[r[1:] for r in day[j]] for day in split], axis=0)
            modes[loc] = {k[0]: dict(hits=int(a), total=int(b), fraction=float(a/max(1, b))) for k,(a,b) in zip(split[0][j], counts)}
        report[name] = modes
    (OUT/'trade_location_audit.json').write_text(json.dumps(report, indent=2))
    for name, modes in report.items():
        print(name, {k: [round(row['fraction'], 3) for row in m.values()] for k,m in modes.items()}, flush=True)


if __name__ == '__main__':
    run()
