"""Compare Y3 tail ranking errors for newly reversed and persistent pairwise forecasts at matched margins."""
import json
import numpy as np
import torch

import y3_risk_rank_validation as history

trial = history.parent
OUT = trial.clock.ART/'y3_pair_crossings'
I, J = np.triu_indices(50, 1)


def tail(p):
    c = trial.curve.settled.historical_geometry()[1]
    return p[:, :, 2] - (c[2, 1]/c[1, 1])*p[:, :, 1]


def diff(p):
    return p[:, I]-p[:, J]


def one(raw, d):
    p = history.references(d, raw)[0]
    y = trial.clock.labels(d).reshape(-1, 50, 3)
    t = np.arange(128 + d%9, len(p)-60, 9)
    q = tail(p)
    now = diff(q[t])
    late = diff(tail(y[t]))
    direction = np.sign(now)
    size = abs(now)
    short = direction*diff(p[t, :, 1]) >= 0
    persist = direction*diff(q[t-32]) >= 0
    # Quantile bins are computed within each live cross section.
    group = np.sum(size[:, :, None]>np.quantile(size, [.25,.5,.75], axis=1).T[:,None], axis=2)
    stats = np.zeros((4,2,2,5))
    for k in range(4):
        for a in range(2):
            for b in range(2):
                keep = (group==k)&(short==a)&(persist==b)&(direction!=0)
                signed = (late*direction)[keep]
                stats[k,a,b] = (len(signed), (signed>0).sum()+.5*(signed==0).sum(), signed.sum(dtype=float),
                                 size[keep].sum(dtype=float), abs(signed).sum(dtype=float))
    return stats


def main():
    OUT.mkdir(exist_ok=True)
    daily = np.stack([one(trial.clock.book.rawday(d),d) for d in range(8,16)])
    def report(g):
        return dict(count=g[...,0].astype(int).tolist(),win=(g[...,1]/g[...,0]).tolist(),
                    payout=(g[...,2]/g[...,0]).tolist(),margin=(g[...,3]/g[...,0]).tolist())
    out = dict(dimensions=['absolute far forecast gap quartile','Y2 direction agrees','same direction 32 ticks earlier'],
               periods=[report(g.sum(0)) for g in (daily[:4],daily[4:])],daily=[report(g) for g in daily])
    (OUT/'results.json').write_text(json.dumps(out,indent=2))
    for g in (daily[:4].sum(0), daily[4:].sum(0)):
        print(np.round(g[...,1]/g[...,0],3), flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    main()
