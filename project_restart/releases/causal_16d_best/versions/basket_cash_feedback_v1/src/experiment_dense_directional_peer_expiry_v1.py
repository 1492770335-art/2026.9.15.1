"""Track peer impulses until the recipient crosses its own price barrier in the same direction."""
from functools import lru_cache
import sys

import numpy as np
import torch
from scipy.ndimage import maximum_filter1d, minimum_filter1d

import prepare_dense_paid_utility_reference_v1 as best


strip, rel, onset = best.model.strip, best.model.rel, best.model.onset
REFERENCE, read_p = best.OUT, best.read_p
OUT = REFERENCE.parent / 'dense_directional_peer_expiry_v1'
ARMS = ('expiry_by_recipient_side', 'synchronized_peer_events', 'network_beyond_shuffled_edges')
HISTORY = 48
PERM = np.random.default_rng(173501).permutation(50)
CONTROLS = 8


def sessions(raw, grid):
    k, coef, bad = strip.physical()
    for a, b in rel.fields.parent.sessions(grid):
        v = raw[a:b]
        ok = ~np.all(v == bad, axis=-1)
        move = np.log1p(k*(v[..., 0].astype(float)-.00136))*coef[2]/k
        move *= ok
        yield a, b, move, ok


def signed_events(move, ok, floor):
    t = np.arange(len(move))
    path = move.cumsum(0)
    sq = np.vstack((np.zeros(50), (move**2).cumsum(0)))
    start = (t-120).clip(0)
    vol = np.sqrt((sq[t]-sq[start]+12*floor**2)/(t-start+12)[:, None])
    # An odd-width filter shifted to its trailing edge uses only the current and preceding prices.
    low = minimum_filter1d(path, 13, axis=0, origin=6, mode='nearest')
    high = maximum_filter1d(path, 13, axis=0, origin=6, mode='nearest')
    strength = np.stack((path-low, high-path), axis=2) / vol[..., None].clip(.0001)
    gaps = np.vstack((np.zeros(50, int), (~ok).cumsum(0)))
    complete = (gaps[t+1] == gaps[(t-12).clip(0)]) & (t >= 12)[:, None]
    hit = (strength > 2.5) & complete[..., None]
    was_hit = np.concatenate((np.zeros_like(hit[:1]), hit[:-1]))
    return hit & ~was_hit


@lru_cache(maxsize=1)
def geometry():
    # Freeze the topology before even the first dates used for readout calibration.
    squares = []
    moments = []
    for d in range(4):
        raw, grid = [onset.settled.load(d, key) for key in ('x', 'time')]
        for _, _, move, ok in sessions(raw, grid):
            squares.append((move**2).sum(0)/ok.sum(0).clip(1))
            n = len(move)//24
            sample = move[:n*24].reshape(n, 24, 50).sum(1)
            sample -= sample.mean(1, keepdims=True)
            moments.append(np.cov(sample.T))
    g = np.mean(moments, axis=0)
    sd = np.sqrt(g.diagonal().clip(1e-8))
    affinity = g/np.outer(sd, sd)
    np.fill_diagonal(affinity, -np.inf)
    idx = np.argsort(affinity, axis=1)[:, -8:]
    w = np.zeros_like(g)
    np.put_along_axis(w, idx, np.take_along_axis(affinity, idx, axis=1).clip(.01), axis=1)
    w /= w.sum(1, keepdims=True)
    return np.sqrt(np.median(squares, axis=0)), w


def features(move, ok, p, floor, edges):
    hit = signed_events(move, ok, floor)
    t = np.arange(len(move))[:, None, None]
    last = np.maximum.accumulate(np.where(hit | ~ok[..., None], t, -1), axis=0)
    start = np.maximum(last+1, t-HISTORY+1).clip(0)
    age = t+1-start
    cols = np.arange(50)[None, :, None]
    sides = np.arange(2)
    lookback = (t-HISTORY+1).clip(0)

    def traces(w):
        pulses = (hit.transpose(0, 2, 1) @ w.T).transpose(0, 2, 1)
        count = np.concatenate((np.zeros_like(pulses[:1]), pulses.cumsum(0)))
        # Coincident donors produce a larger pulse under square weighting.
        def pending(tape):
            return (tape[1:]-tape[start, cols, sides]) / np.sqrt(age.clip(1))
        raw = pending(count)
        sq = np.concatenate((np.zeros_like(pulses[:1]), (pulses**2).cumsum(0)))
        marked = pending(sq)
        unexpired = (count[1:]-count[lookback[:, 0, 0]]) / np.sqrt(np.minimum(t+1, HISTORY))
        def diff(x):
            return x[..., 0]-x[..., 1]
        return np.stack((diff(unexpired), diff(raw), diff(marked)), axis=2)

    wrong = edges[PERM][:, PERM].copy()
    own = hit[..., 0].astype(float)-hit[..., 1]
    age_diff = (age[..., 0]-age[..., 1])/HISTORY
    net = traces(edges)
    random = traces(wrong)
    z = np.concatenate((p, np.stack((own, age_diff, net[..., 0],
                                      random[..., 0], move/floor.clip(.001)), axis=2),
                        net[..., 1:], random[..., 1:]), axis=2)
    return strip.center(z.astype(float))


@lru_cache(maxsize=12)
def gram(d):
    floor, graph = geometry()
    raw, grid, y = [onset.settled.load(d, k) for k in ('x', 'time', 'y')]
    p = read_p(d)
    rows = []
    for a,b,move,ok in sessions(raw,grid):
        i = np.arange(55+d%5, b-a-60, 5)
        x = features(move,ok,p[a:b],floor,graph)[i]
        target = strip.center(y[a+i, :, 1:].astype(float))
        rows.append(np.concatenate((x,target),axis=-1).reshape(-1,14))
    row = np.concatenate(rows)
    return row.T@row/len(row)


def train(end):
    floor, edges = geometry()
    gs = np.stack([gram(d) for d in range(4,end)])
    g = gs.mean(0)[:12,:12]
    units = gs[:,[1,2],[-2,-1]]/gs[:,[1,2],[1,2]].clip(1e-8)
    residual = gs[:,:12,-2:]/units[:,None,:].clip(.2,4)-gs[:,:12,1:3]
    co = np.zeros((12,3,2))
    for j,(idx,control) in enumerate([([8], list(range(CONTROLS))),
                                       ([9], list(range(CONTROLS))+[8]),
                                       ([8,9], list(range(CONTROLS))+[10,11])]):
        proj = np.eye(12)[:,idx]
        nuisance = g[np.ix_(control,control)]
        proj[control] = -np.linalg.solve(nuisance+.05*np.diag(nuisance.diagonal().clip(1e-6)),
                                         g[np.ix_(control,idx)])
        cov = proj.T@g@proj
        cross = proj.T@residual
        mean = cross.mean(0)
        reliability = mean**2/(mean**2+cross.var(0)/len(gs)+1e-10)
        co[:,j] = proj@np.linalg.solve(cov+.5*np.diag(cov.diagonal().clip(1e-7)), mean*reliability)
    return np.r_[floor, edges.ravel(), co.ravel()]


def signals(raw, grid, p, bank):
    floor, edges = bank[:50], bank[50:2550].reshape(50,50)
    co = bank[2550:].reshape(12,6)
    out = np.zeros((*p.shape[:2],3,2))
    for a,b,move,ok in sessions(raw,grid):
        out[a:b] = (features(move,ok,p[a:b],floor,edges)@co).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__ == '__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
