"""Compare two inferred transaction prices to the independently reconstructed side-volume average."""
from functools import lru_cache
import json

import numpy as np

import experiment_execution_volume_clock_v1 as clock
import experiment_y3_inferred_print_impacts_v1 as source

book, trial = source.book, source.trial
OUT = source.events.OUT


@lru_cache(maxsize=1)
def calibration():
    state = source.parameters()
    origins, slopes = [], []
    for day in range(4):
        raw = trial.clock.book.rawday(day)
        mid, width, offset, ok = book.quote.infer(raw, float(state['k']), state['coefficients'])
        buy, sell = raw[:, :, 54]>state['floors'][1], raw[:, :, 55]>state['floors'][2]
        f = raw[:, :, 11].astype(float)
        zero = np.median(f[~(buy|sell)])
        sel = ok & (buy^sell) & (offset!=0) & (mid>10)
        slope = np.median((f[sel]-zero)*state['k']*mid[sel]/offset[sel])
        origins.append(zero); slopes.append(slope)
    with np.load(clock.OUT/'model.npz') as f:
        extra = {k: f[k].copy() for k in ('keys0', 'keys1', 'values0', 'values1')}
    return dict(**state, **extra, average=np.array([np.median(origins), np.median(slopes)]))


def prices(raw, p):
    mid, half, offset, ok = book.quote.infer(raw, float(p['k']), p['coefficients'])
    vols = []
    for j in (0, 1):
        x = raw[:, :, 54+j].astype(float)
        vols.append(np.where(x>p['floors'][1+j], np.interp(x, p[f'keys{j}'], p[f'values{j}']), 0))
    buy, sell = vols
    frac = (buy-sell)/(buy+sell).clip(.0001)
    avg = (raw[:, :, 11].astype(float)-p['average'][0])*mid*p['k']/p['average'][1]
    return mid, half, offset, avg, ok, frac, buy, sell


def run():
    state = calibration()
    records = []
    for day in range(8, 16):
        mid, half, offset, avg, ok, frac, buy, sell = prices(trial.clock.book.rawday(day), state)
        idx = np.r_[0, np.arange(len(mid)-1)]
        stable = ok & ok[idx] & (mid==mid[idx]) & (half==half[idx])
        stable[0] = False
        groups = []
        for sel in (stable & ((buy>0)^(sell>0)), stable & (buy>0) & (sell>0), stable & (buy==0) & (sell==0)):
            # When both quotes stay put, the side-volume average can be obtained without either price field.
            ref = frac[sel]*half[sel]
            a = np.column_stack((avg[sel], offset[sel], np.zeros(sel.sum())))
            diffs = abs(a-ref[:, None])/half[sel, None]
            groups.append(dict(rows=int(sel.sum()), within_tenth=np.mean(diffs<.10, axis=0).tolist(),
                               mean_abs_error=diffs.mean(0).tolist(), median_abs_error=np.median(diffs,0).tolist()))
        records.append(groups)
    report = dict(days=list(range(9, 17)), calibration=state['average'].tolist(),
                  groups=['one side', 'both sides', 'inactive'], estimators=['f011', 'f010', 'midpoint'], daily=records)
    (OUT/'print_average_price_audit.json').write_text(json.dumps(report, indent=2))
    print(report['calibration'])
    for i in range(3):
        print(report['groups'][i], sum(r[i]['rows'] for r in records),
              np.mean([r[i]['within_tenth'] for r in records], axis=0), flush=True)


if __name__ == '__main__':
    run()
