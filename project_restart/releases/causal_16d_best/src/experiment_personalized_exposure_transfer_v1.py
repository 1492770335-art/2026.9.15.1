"""Estimate the historical response to personalized book signals and measure the incumbent's exposure at prediction time."""
import json
import numpy as np

import experiment_within_stock_state_matches_v1 as matches
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, measure

OUT = ART/'personalized_exposure_transfer_v1'
best, book = matches.best, matches.BOOK
WINDOW = 240
PRIOR = 64
MIX = .24


def daily(p, y, s):
    p, y, s = (v.astype(float) for v in (p,y,s))
    pc = p-p.mean((0,1))
    yc = y-y.mean((0,1))
    slope = ((pc*yc).sum((0,1))/(pc*pc).sum((0,1))).clip(.4, 2.5)
    def moment(a):
        return np.mean(s*a, axis=(0,1))
    return np.stack((moment(s), moment(p), moment(y)/slope))


def calibration(hist):
    rows = np.stack(hist)
    var = rows[:,0].mean(0).clip(1e-8)
    # Shrink per-date slopes to limit influence from a handful of transient moves.
    def coefficient(j):
        avg = rows[:,j].mean(0)/var
        each = rows[:,j]/(rows[:,0]+.25*var)
        return (.65*avg+.35*1.25*np.median(each, 0)).clip(-12,12)
    return np.stack((var, coefficient(1), coefficient(2)))


def adjust(p, s, pars, dynamic):
    var, historic, target = pars
    t = np.arange(len(p))+1
    if dynamic:
        def total(x):
            cs = np.vstack((np.zeros((1,3)), x.cumsum(0,dtype=float)))
            return cs[t]-cs[np.maximum(t-WINDOW, 0)]
        num = total((s*p).mean(1))+PRIOR*historic*var
        den = total((s*s).mean(1))+PRIOR*var
        actual = (num/den.clip(1e-8)).clip(-12,12)[:,None,:]
    else:
        actual = historic[None,None]
    delta = MIX*s*(target-actual).clip(-7,7)
    return (p+delta).astype('float32')


def signal(raw, geo, w):
    out = np.empty((*raw.shape[:2],3), np.float32)
    for i in range(min(4, len(raw))):
        t = np.arange(i,len(raw),4)
        out[t] = matches.signal(matches.basis(raw,t,geo),w,geo)
    return out


def load():
    with np.load(OUT/'model.npz') as f:
        geo = {k:f[k] for k in ('center','scale','risk','knots')}
        w, pars, mask = f['w'], f['calibration'], f['mask']
    parent = best.load()
    def apply(raw,p):
        q = parent(raw,p)
        return np.where(mask,adjust(q,signal(raw,geo,w),pars,True),q)
    return apply


def main():
    OUT.mkdir(exist_ok=True)
    history, cal, daily_scores, folds = [], [], [], []
    geo = matches.geometry()
    for end in (8,10,12,14):
        for d in range(len(history),end):
            history.append(matches.statistics(book.rawday(d), labels(d).reshape(-1,50,3),d,geo))
        w = matches.weights(history,geo,1)
        pars = calibration(cal) if cal else None
        group = [[[],[]] for _ in range(3)]
        for d in range(end,end+2):
            raw = book.rawday(d)
            s = signal(raw,geo,w)
            y = labels(d).reshape(-1,50,3)
            for j,name in enumerate(('panel','neural','opening')):
                path = best.refs.OUT/f'{name}_{d}.npy' if j<2 else ART/'opening_oof_cache_v1'/f'day_{d:02d}.npy'
                p = np.load(path).reshape(y.shape)
                if not j:
                    cal.append(daily(p,y,s))
                if pars is None:
                    continue
                for i in (0,1):
                    q = adjust(p,s,pars,i)
                    if d==end and j==0 and i:
                        n = 279
                        short = adjust(p[:n],signal(raw[:n],geo,w),pars,i)
                        np.testing.assert_allclose(short,q[:n],rtol=0,atol=3e-6)
                    row = tuple(a.reshape(-1,3) for a in (p,q,y))
                    group[j][i].append(row)
                    if not j and i:
                        sc = measure([row]); daily_scores.append(np.subtract(sc['candidate'],sc['base']))
        if pars is not None:
            f = [[measure(items) for items in ref] for ref in group]
            folds.append(f)
            print(end+1,[[s['gain'] for s in ref] for ref in f],flush=True)
    gain = np.array([[np.subtract(b['candidate'],b['base']) for a,b in f] for f in folds])
    edge = np.array([[np.subtract(b['candidate'],a['candidate']) for a,b in f] for f in folds])
    keep = ((gain[:,0].min(0)>0)&(gain.mean(0).min(0)>0)&(edge[:,0].mean(0)>0)
            &((np.stack(daily_scores)>0).sum(0)>=4))
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,retained=keep.tolist(),calibration=calibration(cal).tolist()),indent=2))
    if keep.any():
        for d in range(len(history),16):
            history.append(matches.statistics(book.rawday(d),labels(d).reshape(-1,50,3),d,geo))
        np.savez(OUT/'model.npz',**geo,w=matches.weights(history,geo,1),calibration=calibration(cal),mask=keep)
        result = best.walk.replay(load())
        result['gain_over_best'] = result['pooled']['score']-json.loads((best.OUT/'raw_replay.json').read_text())['pooled']['score']
        (OUT/'raw_replay.json').write_text(json.dumps(result,indent=2))
        print(result['pooled'],result['gain_over_best'],flush=True)
        if result['gain_over_best']>0:
            checked = best.walk.replay(load())
            np.testing.assert_allclose(result['pooled']['candidate'],checked['pooled']['candidate'],rtol=0,atol=1e-10)
            checked['gain_over_best'] = result['gain_over_best']
            (OUT/'reload_verification.json').write_text(json.dumps(checked,indent=2))


if __name__=='__main__':
    main()
