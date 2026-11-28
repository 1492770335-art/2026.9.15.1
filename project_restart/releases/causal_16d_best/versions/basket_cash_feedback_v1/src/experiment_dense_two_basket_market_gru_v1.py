"""Learn shared price moves from two disjoint donor panels, supervised by a third recipient panel."""
from functools import lru_cache
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dense_exchange_count_mixture_v1 as source
import prepare_dense_equal_span_event_gru_reference_v1 as best

pre, peer, strip, event, onset, rel = source.pre, source.peer, source.strip, source.event, source.onset, source.rel
REFERENCE, read_p = best.OUT, best.read_p
OUT = REFERENCE.parent/'dense_two_basket_market_gru_v1'
ARMS = ('donor_history', 'confirmed_by_both_donors', 'joint_donor_state')
L = pre.H
C, D = 6, 15
BASKETS = np.random.default_rng(34811).permutation(50).reshape(5, 10)
LAGS = np.array([48, 24, 12, 6, 2, 0])


@torch.inference_mode()
def panel(raw, ok, base, frozen):
    # Apply only the stock-local encoder, keeping the recipient out of both donor panels.
    pieces = []
    for start in range(0, len(raw), 128):
        block = np.zeros((48+128, 50, raw.shape[-1]), np.float32)
        begin = max(0, start-48)
        end = min(start+128, len(raw))
        block[48-(start-begin):48+end-start] = raw[begin:end]
        x = pre.inputs(block)[48:]
        z = torch.tensor(((x-base[:pre.N])/base[pre.N:2*pre.N]).clip(-6, 6),
                         device=pre.DEVICE, dtype=torch.float32)
        pieces.append(frozen.encode(z)[:end-start].cpu().numpy().astype(np.float64))
    tokens = np.concatenate(pieces)
    live = ok[:, BASKETS]
    n = live.sum(2)
    sums = (tokens[:, BASKETS]*live[..., None]).sum(2)
    donors = []
    for i, j in ((1, 2), (3, 4)):
        sums_ij = np.roll(sums, -i, axis=1)+np.roll(sums, -j, axis=1)
        n_ij = np.roll(n, -i, axis=1)+np.roll(n, -j, axis=1)
        donors.append(sums_ij/n_ij.clip(1)[..., None])
    return np.stack(donors, axis=2).astype(np.float32)


class MarketGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.frame = nn.Linear(L, 20)
        self.local = nn.Linear(20, 3)
        self.gru = nn.GRU(20, 16, batch_first=True)
        self.leg = nn.Linear(16, 3, bias=False)
        self.pair = nn.Sequential(nn.Linear(48, 16), nn.Tanh(), nn.Linear(16, 3, bias=False))

    def forward(self, x):
        # time, recipient basket, donor panel, history index, feature
        shape = x.shape[:2]
        z = self.frame(x).tanh()
        frame = self.local(z[..., -1, :]).mean(2)
        _, h = self.gru(z.reshape(-1, len(LAGS), 20))
        left, right = h[0].reshape(*shape, 2, 16).unbind(2)
        a, b = self.leg(left), self.leg(right)
        same = torch.where(a*b>0, a.sign()*torch.minimum(a.abs(), b.abs()), 0.)
        cross = self.pair(torch.cat(((left+right)/2, (left-right).abs(), left*right), dim=-1))
        return frame, (a+b)/2, same, cross


def windows(x, t):
    return x[(t[:, None]-LAGS).clip(0)].transpose(0, 2, 3, 1, 4)


@lru_cache(maxsize=1)
def encoder():
    base = pre.encoder()
    frozen = pre.restore(base)
    xx, yy, mask = [], [], []
    for d in range(4):
        raw, grid = [onset.settled.load(d, k) for k in ('x', 'time')]
        for a, b, m, ok in event.sessions(raw, grid):
            t = np.arange(48+d%4, len(m)-60, 4)
            gaps = np.vstack((np.zeros(50, int), (~ok).cumsum(0)))
            good = (gaps[t+61]==gaps[t])[:, BASKETS]
            path = m.cumsum(0)
            delta = np.stack([path[t+hi]-path[t+lo] for lo, hi in source.BANDS], axis=-1)
            n = good.sum(2)
            yy.append((delta[:, BASKETS]*good[..., None]).sum(2)/n.clip(1)[..., None])
            mask.append(n>=7)
            xx.append(windows(panel(raw[a:b], ok, base, frozen), t))
    x, y, live = map(np.concatenate, (xx, yy, mask))
    mu = x[..., -1, :].mean((0, 1, 2))
    sd = x[..., -1, :].std((0, 1, 2)).clip(.006)
    scale = y[live].std(0).clip(.01)
    xx = torch.as_tensor(((x-mu)/sd).clip(-5, 5), device=pre.DEVICE)
    yy = torch.tensor(y/scale, device=pre.DEVICE, dtype=torch.float32)
    keep = torch.tensor(live[..., None], device=pre.DEVICE)
    torch.manual_seed(31725)
    net = MarketGRU().to(pre.DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.06)
    for _ in range(360):
        ix = torch.randint(len(xx), (48,), device=pre.DEVICE)
        frame, hist, _, pair = net(xx[ix])
        terms = (frame, frame.detach()+hist, (frame+hist).detach()+pair)
        loss = sum((F.smooth_l1_loss(term, yy[ix], beta=1., reduction='none')*keep[ix]).mean() for term in terms)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.)
        opt.step()
    w = nn.utils.parameters_to_vector(net.parameters()).detach().cpu().numpy()
    return np.r_[mu, sd, scale, w]


@torch.inference_mode()
def features(raw, ok, p, t, base, bank):
    net = MarketGRU().double().eval()
    w = torch.tensor(bank[2*L+3:], dtype=torch.float64)
    nn.utils.vector_to_parameters(w, net.parameters())
    net.gru.flatten_parameters()
    x = panel(raw, ok, base, pre.restore(base))
    x = ((x-bank[:L])/bank[L:2*L]).clip(-5, 5)
    result = []
    for start in range(0, len(t), 64):
        batch = t[start:start+64]
        z = torch.tensor(windows(x, batch), dtype=torch.float64)
        z = F.pad(z, (0, 0, 0, 0, 0, 0, 0, 0, 0, 64-len(z)))
        legs = [v.mean(1)[:len(batch)].cpu().numpy()*bank[2*L:2*L+3] for v in net(z)]
        result.append(np.column_stack((p[batch].mean(1, dtype=np.float64), *legs)))
    return np.concatenate(result)


@lru_cache(maxsize=12)
def record(day):
    raw, grid, y = [onset.settled.load(day, k) for k in ('x', 'time', 'y')]
    p = read_p(day)
    xx, yy, pp = [], [], []
    for a, b, m, ok in event.sessions(raw, grid):
        t = np.arange(48+day%4, len(m)-60, 4)
        xx.append(features(raw[a:b], ok, p[a:b], t, pre.encoder(), encoder()))
        yy.append(y[a+t, :, 1:].astype(float))
        pp.append(p[a+t, :, 1:].astype(float))
    x, y, p = map(np.concatenate, (xx, yy, pp))
    x -= x.mean(0)
    y -= y.mean((0, 1))
    p -= p.mean((0, 1))
    slope = ((p*y).sum((0, 1))/(p*p).sum((0, 1))).clip(.2, 4)
    return x.T@x/len(x), x.T@(y/slope-p).mean(1)/len(x)


def train(end):
    gg, ee = map(np.stack, zip(*(record(d) for d in range(4, end))))
    g = gg.mean(0)
    co = []
    for ix in np.array_split(np.arange(C, D), 3):
        v = np.eye(D)[:, ix]
        v[:C] = -peer.base.solve(g[:C, :C], g[:C, ix], .07)
        projected = v.T@ee
        avg = projected.mean(0)
        reliable = avg**2/(avg**2+2*projected.var(0)/len(gg)+1e-14)
        co.append(v@peer.base.solve(v.T@g@v, avg*reliable, .9))
    base = pre.encoder()
    return np.r_[len(base), base, encoder(), np.stack(co, axis=1).ravel()]


def signals(raw, grid, p, params):
    cut = 1+int(params[0])
    out = np.zeros((*p.shape[:2], 3, 2))
    for a, b, m, ok in event.sessions(raw, grid):
        x = features(raw[a:b], ok, p[a:b], np.arange(len(m)), params[1:cut], params[cut:-D*6])
        out[a:b] = (x@params[-D*6:].reshape(D, 6)).reshape(b-a, 1, 3, 2)
    return out


def predict(raw, grid, p, params):
    return strip.evaluation.apply(sys.modules[__name__], raw, grid, p, params)


if __name__ == '__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
