"""Route matched transaction pressure to assets with a replicated short- or long-delay price response."""
from functools import lru_cache
import sys

import numpy as np
import torch

import y3_trade_spell_references as history

spell, trial, study = history.incumbent, history.trial, history.study
prints, book = spell.model, spell.book
OUT = trial.clock.ART/'y3_transaction_pressure_network_v1'
ARMS = ('own signed flow', 'co-trading peers', 'immediate price response', 'delayed price response',
        'delayed response after common component', 'permuted recipients', 'late response and own price',
        'delayed response conditional on weak own flow')
CANDIDATES = (1, 3, 4, 6, 7)
CONTROLS = (0, 2, 5)


def flows(raw, state):
    _, _, actual, old, _, _, size = prints.prints(raw, state)
    signed = size*(actual==old)[:, :, None]
    cs = np.concatenate((np.zeros_like(signed[:1]), signed.cumsum(0)))
    return cs


def flow_view(cs, t, width):
    e = t+1
    return (cs[e]-cs[(e-width).clip(0)])/np.sqrt(width)


@lru_cache(maxsize=16)
def moments(day):
    raw = trial.clock.book.rawday(day)
    state = prints.parameters()
    cs = flows(raw, state)
    t = np.arange(200+day%5, len(raw)-60, 5)
    z = flow_view(cs, t, 20).mean(2)
    k = float(state['k'])
    cum = np.log1p(k*(raw[:, :, 0].astype(float)-float(np.float16(.00136)))).cumsum(0)/k
    def normalized(x):
        x = trial.clock.center(x)
        x -= x.mean(0)
        return x/x.std(0).clip(.03)
    z = normalized(z)
    rates = [normalized(cum[t+hi]-cum[t+lo]) for lo, hi in ((0, 5), (20, 40), (40, 60))]
    return np.stack([z.T@v/len(z) for v in (z, *rates)])


def direction(a, b, k):
    # Retain only directionally replicated cross-stock associations, estimated on disjoint dates.
    strength = np.where(a*b>0, np.sign(a)*np.minimum(abs(a), abs(b)), 0.)
    np.fill_diagonal(strength, 0.)
    if k:
        u, s, v = np.linalg.svd(strength, full_matrices=False)
        strength = (u[:, :k]*s[:k])@v[:k]
        np.fill_diagonal(strength, 0.)
    return strength/np.maximum(abs(strength).sum(0), .15)


@lru_cache(maxsize=7)
def params(end):
    stats = np.stack([moments(d) for d in range(end)])
    a, b = stats[::2].mean(0), stats[1::2].mean(0)
    same, near = direction(a[0], b[0], 0), direction(a[1], b[1], 4)
    late = direction(a[2:].mean(0), b[2:].mean(0), 4)
    u, s, v = np.linalg.svd(late, full_matrices=False)
    residual = late-s[0]*np.outer(u[:, 0], v[0])
    np.fill_diagonal(residual, 0.)
    perm = np.random.default_rng(11039).permutation(50)
    return dict(**prints.parameters(), graphs=np.stack((same, near, late, residual, late[np.ix_(perm,perm)])).astype('float32'))


def design_with(raw, p, t, state):
    cs = flows(raw, state)
    own = trial.clock.center(np.concatenate([flow_view(cs,t,w) for w in (5,20,60,180)], axis=2)).astype('float32')
    send = own.transpose(0, 2, 1)
    routed = [(send@g).transpose(0, 2, 1) for g in state['graphs']]
    price = np.tanh(p[t, :, 0:1]/.35)
    conflict = routed[2]*np.tanh(own)
    parts = (p[t], own, *routed, routed[2]*price, routed[2]/(1+abs(own)), conflict)
    return trial.clock.center(np.concatenate(parts, 2)).astype('float32')


def historical_design(day):
    state = params(max(4, day-day%2))
    return lambda raw,p,t: design_with(raw,p,t,state)


def fit(rows):
    own, peers, near, late, res, shuffled, gate, quiet, overlap = [np.arange(3+8*i, 11+8*i) for i in range(9)]
    specs = [(own,3),(peers,11),(near,11),(late,11),(res,11),(shuffled,11),
             (np.r_[late,gate],11),(np.r_[quiet,overlap],11)]
    return dict(**book.prev.gram(rows,[(g,n,.35) for g,n in specs]),**params(len(rows)+4))


changes = book.changes


def correction():
    with np.load(OUT/'model.npz') as f:
        state=dict(f)
    def update(raw,p):
        return changes(design_with(raw,p,np.arange(len(raw)),state),state)[int(state['pick'])]
    return update


def layer():
    return study.layer(sys.modules[__name__], spell)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
