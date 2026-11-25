"""Estimate common top-level depth from old-price fills on either side and test held-date size reconstruction."""
from functools import lru_cache
import json

import numpy as np
from scipy.stats import rankdata

import audit_y3_depleted_queue_fields as field

trial, OUT = field.trial, field.OUT


@lru_cache(maxsize=1)
def limits():
    lower, upper = [], []
    for day in range(4):
        z = trial.clock.book.rawday(day)[:,:,21]
        lower.append(float(z.min())); upper.append(float(z.max()))
    return min(lower), max(upper)


def split_fraction(x, direction, bounds):
    lo, hi = bounds
    bid = ((x-lo)/(hi-lo)).clip(.003,.997)
    return np.where(direction>0, 1-bid, bid)


@lru_cache(maxsize=16)
def records(day):
    x, _, _, code, sign, amount, total, _ = field.samples(day)
    share = split_fraction(x[:,21], sign, limits())
    valid = (share>.015)&(amount>.0002)
    return x[valid], code[valid], sign[valid], np.log(amount[valid]/share[valid]), amount[valid], share[valid]


def pool(days):
    return [np.concatenate(v) for v in zip(*(records(d) for d in days))]


def corr(a,b):
    return float(np.corrcoef(rankdata(a),rankdata(b))[0,1])


@lru_cache(maxsize=1)
def scales():
    x, code, sign, logdepth, *_ = pool(range(4))
    all_stats = []
    for parity in (0,1):
        xx, cc, ss, yy, *_ = pool(range(parity,4,2))
        parts = []
        for stock in range(50):
            for side in (1,-1):
                sel=(cc==stock)&(ss==side)
                if sel.sum()>=12:
                    parts.append(field.corr_matrix(xx[sel],yy[sel]))
        all_stats.append(np.mean(parts,0))
    a,b=all_stats
    score=np.where(a*b>0,np.minimum(abs(a),abs(b)),0)
    order=np.argsort(-score)[:8]
    # Empirical monotone-bin prediction; duplicate cutpoints are handled by interpolation.
    knots, values = [], []
    for j in order:
        xs, ys = [], []
        for idx in np.array_split(np.argsort(x[:,j],kind='stable'),32):
            xs.append(float(np.mean(x[idx,j]))); ys.append(np.median(logdepth[idx]))
        knots.append(xs); values.append(ys)
    return dict(qfields=order, knots=np.asarray(knots), logs=np.asarray(values), top_limits=np.asarray(limits()),
                prior=np.array(np.median(logdepth)), discovery=score[order])


def predict(raw, params, i):
    return np.interp(raw[...,params['qfields'][i]], params['knots'][i], params['logs'][i])


def run():
    params=scales()
    x, code, sign, logdepth, amount, share=pool(range(8,16))
    scores=[]
    for i,j in enumerate(params['qfields']):
        z=predict(x,params,i)
        for_group=[]
        for s in range(50):
            m=code==s
            if m.sum()>15:for_group.append(corr(z[m],logdepth[m]))
        raw_amount=np.exp(z)*share
        scores.append(dict(field=int(j),discovery=float(params['discovery'][i]),rank_correlation=float(np.mean(for_group)),
                           side_size_rank=corr(raw_amount,amount),median_log_error=float(np.median(abs(z-logdepth)))))
    result=dict(discovery=[1,4],replication=[9,16],events=len(x),constant_median_log_error=float(np.median(abs(params['prior']-logdepth))),
                only_imbalance_size_rank=corr(share,amount),fields=scores)
    (OUT/'total_scale.json').write_text(json.dumps(result,indent=2))
    print(result,flush=True)


if __name__=='__main__':
    run()
