"""Estimate the change in a book signal's response over a recurring 20-tick activity cycle."""
import json
import numpy as np

import experiment_external_event_prior_v1 as incumbent
import experiment_pairwise_book_tree_v1 as book
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, corr

OUT = ART / 'periodic_book_response_v1'
PERIOD = 20
BOUNDARY = 2401
RATIO = .160943 / .272164
SHRINK = .30


def center(x):
    return x - x.mean(1, keepdims=True)


def clock_geometry():
    curves = []
    for day in range(8):
        raw = book.rawday(day)
        activity = (abs(raw[:, :, 0].astype(float)-.00136) > .022).mean(1)
        ticks = np.arange(len(raw))
        parts = []
        for mask in ((ticks > 200) & (ticks < 2300), (ticks > 2500) & (ticks < 4500)):
            parts.append([activity[mask & (ticks % PERIOD == k)].mean() for k in range(PERIOD)])
        curves.append(parts)
    am, pm = np.mean(curves, axis=0)
    choices = np.arange(-3, 4)
    shift = choices[np.argmax([np.corrcoef(am, np.roll(pm, -s))[0, 1] for s in choices])]
    return int(shift), np.asarray(curves)


def phase(t, shift):
    return (t - (t >= BOUNDARY) * shift) % PERIOD


def features(raw, t):
    def f(k):
        return np.arcsinh(raw[:, :, k].astype(np.float32))
    r = raw[:, :, 0].astype(np.float32) - .00136
    x = np.stack((f(0), f(11), f(14), f(21), f(26)-f(38), f(41)-f(42),
                  f(50)-f(51), f(54)-f(55), f(77)-f(78)), axis=-1)
    x = np.tanh(x)
    def average(a, w):
        acc = np.concatenate((np.zeros_like(a[:1], dtype=float), a.cumsum(0, dtype=float)))
        end = t+1
        left = (end-w).clip(0)
        return ((acc[end]-acc[left])/(end-left).reshape((-1,)+(1,)*(a.ndim-1))).astype('float32')
    rv = np.sqrt(.25+average(r*r, 80))
    small = x[:, :, [0, 1, 5, 7, 8]]
    now = x[t]
    slow = average(small, 40)
    b = np.concatenate((now, average(small, 6)-slow, slow,
                        np.tanh(average(r, 20)*4/rv)[:, :, None],
                        now[:, :, [1, 4, 5, 7, 8]] * np.tanh(np.log(rv))[:, :, None]), axis=2)
    return np.column_stack((np.ones(len(t)), b.mean(1))), center(b)


def statistics(day, shift):
    raw = book.rawday(day)
    ticks = np.arange(day % 3, len(raw)-60, 3)
    y = labels(day).reshape(-1, 50, 3)[ticks].astype(float)
    y[:, :, 2] -= RATIO * y[:, :, 1]
    m = y.mean(1)
    cycle = phase(ticks, shift)
    out = []
    for x, outcome in zip(features(raw, ticks), (m, y-m[:, None])):
        rows = []
        for k in range(PERIOD):
            mask = cycle == k
            a = x[mask].reshape(-1, x.shape[-1]).astype(float)
            b = outcome[mask].reshape(-1, 3)
            rows.append((a.T@a/len(a), a.T@b/len(a)))
        out.append(tuple(np.stack(c) for c in zip(*rows)))
    return out


def weights(history):
    both = []
    for side in zip(*history):
        gram, rhs = (np.mean(v, axis=0) for v in zip(*side))
        g, r = gram.mean(0), rhs.mean(0)
        reg = .4*np.diag(g.diagonal().clip(.002)) + .001*np.eye(len(g))
        pooled = np.linalg.solve(g+reg, r)
        # Fit only the phase-dependent deviation from the pooled response.
        residual = rhs - gram @ pooled
        def smooth(x):
            return sum(c*np.roll(x, d, axis=0) for d,c in ((-2,.1),(-1,.2),(0,.4),(1,.2),(2,.1)))
        w = np.linalg.solve(smooth(gram)+g+reg, smooth(residual))
        both.append(w-w.mean(0))
    return both


def changes(raw, t, shift, w, offset=0):
    clock = (phase(t, shift)+offset)%PERIOD
    market, local = features(raw,t)
    m = np.einsum('ti,tij->tj',market,w[0][clock])[:, None]
    r = local @ w[1][clock]
    return [SHRINK*m, SHRINK*center(r)]


def deltas(rows):
    p,y,z = [np.concatenate(col).reshape(-1,3) for col in zip(*rows)]
    return (corr(p+z,y)-corr(p,y)).tolist()


def screen():
    shift, curves = clock_geometry()
    history = []
    folds, daily, direction = [], [], []
    for end in (8,10,12,14):
        history.extend(statistics(d,shift) for d in range(len(history),end))
        w = weights(history)
        groups = [[[] for _ in range(2)] for _ in range(2)]
        control = [[],[]]
        for d in (end,end+1):
            raw = book.rawday(d)
            t = np.arange(d%3,len(raw),3)
            y = labels(d).reshape(-1,50,3)[t]
            changes_true = changes(raw,t,shift,w)
            changes_shifted = changes(raw,t,shift,w,5)
            if d == end:
                for a,b in zip(changes_true,changes(raw[:233],t[t<233],shift,w)):
                    np.testing.assert_allclose(a[t<233],b,atol=2e-6,rtol=0)
            for j, ref in enumerate((incumbent.fi.peer.reference, incumbent.parent.fixed.earlier.early.baseline)):
                p = ref(d).reshape(-1,50,3)[t]
                for s,z in enumerate(changes_true):
                    row = (p,y,np.broadcast_to(z,y.shape))
                    groups[j][s].append(row)
                    if j == 0:
                        control[s].append((p,y,np.broadcast_to(changes_shifted[s],y.shape)))
            daily.append([deltas(rows[-1:]) for rows in groups[0]])
        folds.append([[deltas(rows) for rows in arms] for arms in groups])
        direction.append([deltas(rows) for rows in control])
        print('window',end+1,folds[-1][0],flush=True)
    f = np.array(folds)
    keep = (f.min(axis=(0,1)) > 0) & ((np.array(daily)>0).sum(0)>=5)
    keep &= (f[:,0]-np.array(direction)).mean(0)>0
    report = dict(period=PERIOD,shift=shift,activity_profile=curves.mean(0).tolist(),
                  folds=folds,shifted_control=direction,daily=daily,keep=keep.tolist())
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    if keep.any():
        history.extend(statistics(d,shift) for d in (14,15))
        w=weights(history)
        np.savez(OUT/'model.npz',m=w[0],r=w[1],shift=shift,keep=keep)
    return keep.any()


def load():
    model=incumbent.load()
    with np.load(OUT/'model.npz') as file:
        w, shift, keep = [file['m'],file['r']],int(file['shift']),file['keep']
    def apply(raw,p):
        q=model(raw,p).copy()
        for v,k in zip(changes(raw,np.arange(len(raw)),shift,w),keep):
            q+=v*k
        return q
    return apply


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():
        incumbent.parent.fixed.votes.follow.replay_candidate(OUT,load,baseline=incumbent.OUT)
