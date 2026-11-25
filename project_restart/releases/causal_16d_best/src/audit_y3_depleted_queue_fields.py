"""Search for a leading quote-size field using volume inferred to have traded at a depleted quote."""
from functools import lru_cache
import json

import numpy as np
from scipy.stats import rankdata

import audit_y3_print_vs_average_price as tape

trial, book = tape.trial, tape.book
OUT = trial.clock.ART/'y3_depleted_queue_audit'


def samples(day):
    raw = trial.clock.book.rawday(day)
    pars = tape.calibration()
    mid, width, last, avg, valid, frac, buy, sell = tape.prices(raw, pars)
    one = (buy>0)^(sell>0)
    direction = np.sign(buy-sell)
    prev = np.r_[0,np.arange(len(mid)-1)]
    old = mid[prev]+direction*width[prev]
    new = mid+direction*width
    shift = direction*(new-old)
    part = direction*(mid-old+avg)/np.where(shift>0,shift,1)
    # If executions occur at just the two adjacent prices, their average identifies their volume split.
    ok = (one & valid & valid[prev] & (shift==2) & (mid+last==new) & (part>.03) & (part<.97))
    ok[0] = False
    at, code = np.nonzero(ok)
    total = buy[at,code]+sell[at,code]
    paid_old = total*(1-part[at,code])
    return (raw[prev[at],code,:83].astype('float32'), raw[at,code,:83].astype('float32'),
            raw[(at-7).clip(0),code,:83].astype('float32'), code, direction[at,code],
            paid_old, total, at)


def corr_matrix(z, y):
    x = rankdata(z, axis=0)
    v = rankdata(y)
    x -= x.mean(0); v -= v.mean()
    return x.T@v/np.sqrt((x*x).sum(0).clip(1)*(v@v))


@lru_cache(maxsize=16)
def statistics(day):
    pre, post, stale, code, sign, amount, total, at = samples(day)
    dcor, count = [], []
    for side in (1, -1):
        records = []
        n = 0
        for s in range(50):
            rows = (code==s)&(sign==side)
            if rows.sum()<12: continue
            grids = [corr_matrix(x[rows],amount[rows]) for x in (pre, post, stale)]
            grids.append(corr_matrix(pre[rows],total[rows]))
            records.append(grids)
            n += int(rows.sum())
        dcor.append(np.mean(records,0) if records else np.zeros((4,pre.shape[-1])))
        count.append(n)
    return np.stack(dcor), np.array(count), len(code)


def run():
    OUT.mkdir(exist_ok=True)
    stats = [statistics(d) for d in range(16)]
    early = np.mean([s[0] for s in stats[:4]],0)
    later = np.mean([s[0] for s in stats[8:]],0)
    odd = np.mean([s[0] for s in stats[:4:2]],0)
    even = np.mean([s[0] for s in stats[1:4:2]],0)
    reports = []
    for side in range(2):
        rep = np.where(odd[side,0]*even[side,0]>0,np.minimum(abs(odd[side,0]),abs(even[side,0])),0)
        records = []
        for j in np.argsort(-rep)[:8]:
            records.append(dict(field=int(j), early=float(early[side,0,j]), later=float(later[side,0,j]),
                other_side=float(later[1-side,0,j]), after=float(later[side,1,j]), stale=float(later[side,2,j]),
                total_volume=float(later[side,3,j])))
        reports.append(records)
    result = dict(discovery_days=[1,4], replication_days=[9,16],
        events=int(sum(s[2] for s in stats[8:])), eligible_by_side=np.sum([s[1] for s in stats[8:]],0).tolist(),
        sides=['buy','sell'], ranked=reports)
    (OUT/'field_rank.json').write_text(json.dumps(result,indent=2))
    print(result['events'],result['eligible_by_side'])
    for rec in reports:
        print([(r['field'],round(r['early'],3),round(r['later'],3),round(r['other_side'],3),
                round(r['after'],3),round(r['stale'],3)) for r in rec],flush=True)


if __name__=='__main__':
    run()
