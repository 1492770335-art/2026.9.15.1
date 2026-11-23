"""Attribute internal rolling forecast errors to market and relative returns before selecting a model."""
import json
import numpy as np
import experiment_peer_catchup_after_pulse_v1 as peer
from experiment_cross_date_response_shapes_v1 import SCALE

OUT = peer.ART/'y3_error_components_v1'


def components(y, p):
    market_y, market_p = (u.mean(1, keepdims=True) for u in (y, p))
    mse = float(((y-p)**2).mean())
    norm = y.std()*p.std()
    report = {}
    for name, a, b in (('market', market_y, market_p), ('relative', y-market_y, p-market_p)):
        err = a-b
        cov = np.mean((a-a.mean())*(b-b.mean()))
        report[name] = dict(mse=float((err**2).mean()), mse_fraction=float((err**2).mean()/mse),
                            corr=float(cov/(a.std()*b.std())), full_ic_contribution=float(cov/norm),
                            label_variance_fraction=float(a.var()/y.var()), slope=float(cov/b.var()))
    assert abs(sum(v['mse_fraction'] for v in report.values())-1)<1e-9
    report['full_ic'] = float(np.corrcoef(y.ravel(), p.ravel())[0,1])
    assert abs(report['full_ic']-sum(report[n]['full_ic_contribution'] for n in ('market','relative')))<1e-9
    return report


def inspect(y, p):
    ratio = SCALE[2]/SCALE[1]
    return {k: components(a, b) for k, a, b in (
        ('y3', y[:,:,2], p[:,:,2]), ('first20', ratio*y[:,:,1], ratio*p[:,:,1]),
        ('last40', y[:,:,2]-ratio*y[:,:,1], p[:,:,2]-ratio*p[:,:,1]))}


def main():
    OUT.mkdir(exist_ok=True)
    rows, daily = [], {}
    for d in range(8, 16):
        p = peer.reference(d).astype(float)
        y = peer.labels(d).reshape(p.shape).astype(float)
        daily[d+1] = inspect(y, p)
        rows.append((y, p))
    full = inspect(*(np.concatenate(v) for v in zip(*rows)))
    report = dict(pooled=full, daily=daily, rows=sum(y.size//3 for y,p in rows), reference='peer.reference',
                  dates=list(range(9,17)))
    (OUT/'results.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(full))
    for k,r in daily.items():
        print(k,[round(r['y3'][n]['corr'],4) for n in ('market','relative')])

if __name__=='__main__':
    main()
