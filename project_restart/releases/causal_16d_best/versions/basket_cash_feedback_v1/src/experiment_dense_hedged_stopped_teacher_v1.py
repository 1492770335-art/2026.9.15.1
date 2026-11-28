"""Train each stopped-shock student on portfolios with zero current exposure to the three issued forecasts."""
from functools import lru_cache
import sys

import numpy as np
import torch

import experiment_dense_stopped_shock_teacher_v1 as teacher

pre, strip, event, onset, rel = teacher.pre, teacher.strip, teacher.event, teacher.onset, teacher.rel
REFERENCE, read_p, C, D = teacher.REFERENCE, teacher.read_p, teacher.C, teacher.D
OUT = REFERENCE.parent/'dense_hedged_stopped_teacher_v1'
ARMS = teacher.ARMS


def hedge(x, ok):
    obs = np.concatenate((np.ones((*x.shape[:2], 1)), x[..., :C]), axis=-1)
    weighted = obs*ok[..., None]
    xt = weighted.swapaxes(1, 2)
    gram = xt@weighted
    beta = np.linalg.pinv(gram, rcond=1e-10, hermitian=True)@(xt@x[..., C:])
    f = (x[..., C:]-obs@beta)*ok[..., None]
    return np.concatenate((x[..., :C], f), axis=-1)


@lru_cache(maxsize=12)
def moments(day):
    return teacher.calculate_moments(day, hedge)


def train(end):
    gs = np.stack([moments(d) for d in range(4, end)])
    slope = (gs[:, [1,2], [D,D+1]]/gs[:, [1,2], [1,2]]).clip(.2, 4)
    g = gs[:, C:D, C:D].mean(0)
    co = np.zeros((D, 3, 2))
    for a in range(3):
        r = gs[:, C:D, D+2*a:D+2*a+2]/slope[:, None]-gs[:, C:D, 1:3]
        avg = r.mean(0)
        shrink = avg**2/(avg**2+3*r.var(0)/len(r)+1e-14)
        co[C:, a] = pre.peer.base.solve(g, avg*shrink, .85)
    return np.r_[pre.encoder(), co.ravel()]


def signals(raw, grid, p, params):
    base = params[:-D*6]
    net = pre.restore(base)
    out = np.zeros((*p.shape[:2], 3, 2))
    for a,b,_,ok in event.sessions(raw,grid):
        x = hedge(pre.views(raw[a:b], p[a:b], base, net), ok)
        out[a:b] = (x@params[-D*6:].reshape(D, 6)).reshape(b-a, 50, 3, 2)
    return out


def predict(raw, grid, p, params):
    return strip.evaluation.apply(sys.modules[__name__], raw, grid, p, params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
