"""Compress each stock's last price revisits into event tokens for a Y3-only temporal readout."""
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import y3_quote_memory_references as history

model, trial, study = history.incumbent, history.trial, history.study
OUT = trial.clock.ART/'y3_quote_event_tokens_v1'
ARMS = ('equal-weight event pool', 'age-conditioned event attention', 'attention and tail supervision',
        'tail-only attention', 'other stock event ages', 'event contents without temporal weighting',
        'gain from event attention')
CANDIDATES = (1, 2, 3, 6)
CONTROLS = (0, 4, 5)


def design_with(raw, p, t, state):
    mid, width, _, ok = model.quote.infer(raw, float(state['k']), state['coefficients'])
    now = np.arange(len(raw))[:, None]
    stock = np.arange(50)[None, :]
    parts = []
    for side, level in enumerate((mid-width, mid+width)):
        old = model.previous_visits(level, ok)
        age = now-old
        revisit = ok & (old >= 0) & (age > 1) & (age <= 240)
        recent = np.maximum.accumulate(np.where(revisit, now, -1), axis=0)
        x = np.arcsinh(raw[:, :, state['sides'][side]].astype('float32'))
        event = recent[t]
        for _ in range(4):
            avail = (event >= 0) & (t[:, None]-event <= 240)
            tick = event.clip(0)
            previous = old[tick, stock].clip(0)
            delta = x[tick, stock]-x[previous, stock]
            shape = np.stack((np.log1p(tick-previous), np.log1p(t[:, None]-tick), avail), axis=2)
            parts.append((np.concatenate((delta, shape), axis=2)*avail[:, :, None]).astype('float32'))
            event = np.where(event>0, recent[(event-1).clip(0), stock], -1)
    return np.concatenate((p[t], *parts), axis=2).astype('float32')


def design(raw, p, t):
    return design_with(raw, p, t, model.parameters())


def training_error(p, y, t):
    coeff = trial.curve.settled.historical_geometry()[1]
    ratio = coeff[2, 1]/coeff[1, 1]
    actual = y[t, :, 2]-ratio*y[t, :, 1]-(p[t, :, 2]-ratio*p[t, :, 1])
    return trial.clock.center(np.stack((study.error(p, y, t), actual), axis=2)).astype('float32')


class EventPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.content = nn.Sequential(nn.Linear(5, 16), nn.SiLU(), nn.Linear(16, 12))
        self.query = nn.Linear(3, 12)
        self.age = nn.Sequential(nn.Linear(2, 8), nn.SiLU(), nn.Linear(8, 1))
        self.read = nn.Sequential(nn.Linear(25, 16), nn.SiLU(), nn.Linear(16, 2))

    def forward(self, x):
        p = x[:, :, :3]
        events = x[:, :, 3:].reshape(len(x), 50, 2, 4, 7)
        v = self.content(torch.cat((events[..., :4], events[..., 4:5]/4), -1))
        actual = events[..., -1:]
        timing = events[..., 4:6]/4
        q = torch.tanh(self.query(p))[:, :, None, None, :]
        logits = (v*q).sum(-1, keepdim=True)/3.5
        def decode(weights):
            weights = weights*actual
            pool = (v*weights).sum(3)/weights.sum(3).clamp_min(.1)
            coverage = actual.mean((2, 3))
            z = self.read(torch.cat((pool.flatten(2), coverage), 2))
            return z-z.mean(1, keepdim=True)
        def attend(score):
            return decode(torch.exp(score.clamp(-5, 5)))
        simple = decode(torch.ones_like(actual))
        ordered = attend(logits+self.age(timing))
        wrong = attend(logits+self.age(timing.roll(17, 1)))
        content = attend(logits)
        return simple, ordered, wrong, content


def fit(rows):
    torch.manual_seed(173331)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = EventPool().to(device)
    # Preserve uniform weight per date when two chronological references are available.
    weights = np.concatenate([np.full(len(x), 1/len(x)) for x, _ in rows])
    x, y = [torch.from_numpy(np.concatenate(v)).to(device) for v in zip(*rows)]
    w = torch.as_tensor(weights/weights.mean(), device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(net.parameters(), lr=.0011, weight_decay=.07)
    for epoch in range(8):
        if epoch == 5:
            opt.param_groups[0]['lr'] *= .35
        for ids in torch.randperm(len(x), device=device).split(40):
            target = y[ids]
            preds = net(x[ids])
            loss = sum((w[ids]*F.smooth_l1_loss(z, target, beta=.6, reduction='none').mean((1, 2))).mean()
                       for z in preds)/len(preds)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), .6)
            opt.step()
    state = {k: v.detach().cpu().numpy() for k, v in net.state_dict().items()}
    return dict(**state, **model.parameters())


@torch.inference_mode()
def changes(x, state):
    # The small readout runs on CPU for reproducible batch and prefix evaluation.
    net = EventPool().eval()
    net.load_state_dict({k: torch.from_numpy(state[k].copy()) for k in net.state_dict()})
    outputs = []
    for batch in torch.from_numpy(x).split(32):
        simple, joint, wrong, content = net(batch)
        a, b = joint.unbind(-1)
        outputs.append(torch.stack((simple[:, :, 0], a, .6*a+.4*b, b, wrong[:, :, 0], content[:, :, 0],
                                    a-simple[:, :, 0]), -1).numpy())
    return list(np.moveaxis(.24*np.concatenate(outputs), 2, 0))


def correction():
    with np.load(OUT/'model.npz') as f:
        state = dict(f)
    def update(raw, p):
        return changes(design_with(raw, p, np.arange(len(raw)), state), state)[int(state['pick'])]
    return update


def layer():
    return study.layer(sys.modules[__name__], model)


if __name__ == '__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__], reference=history)
