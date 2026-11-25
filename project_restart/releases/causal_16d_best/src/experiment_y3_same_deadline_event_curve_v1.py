"""Compare successive forecasts of the same price endpoint around each stock's last price change."""
import sys

import numpy as np
import torch

import y3_full_residual_trial as study

screen, trial, parent = study.screen, study.trial, study.parent
OUT = trial.clock.ART / 'y3_same_deadline_event_curve_v1'
ARMS = ('fixed vintage', 'before price event', 'immediately after price event', 'event revision',
        'revision term slope', 'mismatched deadline', 'other stock event clock', 'before and after', 'half revision')
CANDIDATES = (1, 2, 3, 4, 7, 8)
CONTROLS = (0, 5, 6)
ENDS = (5, 20, 40)


def curve(levels, clock, h):
    stock = np.arange(50)[None, :]
    a, b, c = np.moveaxis(levels[clock, stock], 2, 0)
    return np.where(h <= 5, a*h/5, np.where(h <= 20, a+(b-a)*(h-5)/15, b+(c-b)*(h-20)/40))


def design(raw, p, t):
    k, coef, _ = trial.curve.settled.historical_geometry()
    levels = np.log1p(k*((p-coef[:, 0])/coef[:, 1]).clip(-.95/k, None))
    price = np.log1p(k*(raw[:, :, 0].astype(float)-.00136)).cumsum(0)
    event = raw[:, :, 0] != np.float16(.00136)
    times = np.arange(len(raw))[:, None]
    last = np.maximum.accumulate(np.where(event, times, -1), axis=0)[t]
    stock = np.arange(50)[None, :]
    now = np.broadcast_to(t[:, None], last.shape)

    def view(origin, h, matched=True):
        age = now-origin
        valid = (origin >= 0) & (age+h <= 60)
        safe = origin.clip(0)
        # Both forecasts refer to the price at now+h, measured from the current spot.
        prior = curve(levels, safe, h+age if matched else h) + price[safe, stock] - price[now, stock]
        return np.where(valid, (prior-curve(levels, now, h))*coef[2, 1]/k, 0)

    chunks = [[] for _ in range(7)]
    swap = np.roll(last, 17, axis=1)
    for h in ENDS:
        pre, post = view(last-1, h), view(last, h)
        revision = post-pre
        wrong = view(last, h, False)-view(last-1, h, False)
        other = view(swap, h)-view(swap-1, h)
        recent = revision*np.exp(-(now-last).clip(0)/12)
        for chunk, value in zip(chunks, (view(now-3, h), pre, post, revision, wrong, other, recent)):
            chunk.append(value)
    arrays = [np.stack(chunk, axis=2) for chunk in chunks]
    slopes = np.diff(arrays[3], axis=2)
    return trial.clock.center(np.concatenate((p[t], *arrays, slopes), axis=2)).astype('float32')


def gram(rows, specs):
    moments = []
    for x, y in rows:
        a, r = x.reshape(-1, x.shape[-1]), y.ravel()
        moments.append((a.T@a/len(a), a.T@r/len(a)))
    return dict(w=screen.projected_readouts(moments, specs))


def fit(rows):
    fixed, pre, post, rev, wrong, other, recent = [np.arange(3+i*3, 6+i*3) for i in range(7)]
    slopes = np.arange(24, 26)
    return gram(rows, [(fixed, 3, .4), (pre, 6, .4), (post, 6, .4), (rev, 6, .4), (slopes, 6, .4),
                       (wrong, 6, .4), (other, 6, .4), (np.r_[pre, post, recent], 6, .4), (rev, 6, .2)])


def changes(x, state):
    return list(np.moveaxis(x@state['w'], 2, 0))


def correction():
    return study.frozen(sys.modules[__name__])


def layer():
    return study.layer(sys.modules[__name__], parent.incumbent)


if __name__ == '__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__])
