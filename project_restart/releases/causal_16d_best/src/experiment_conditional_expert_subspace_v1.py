"""Predict stock-relative residuals from expert directions outside the three existing forecasts."""
import json
import numpy as np
import torch
from torch import nn

import experiment_rolling_expert_reference_v1 as rolling
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, measure

OUT = ART / 'conditional_expert_subspace_v1'
N = 8
CONTEXT = 3 * (N + 3) + 1


def inputs(p, experts, t):
    q = p[t].astype(float)
    x = q - q.mean(1, keepdims=True)
    f = np.stack([e[t] for e in experts], axis=-1)
    f -= f.mean(1, keepdims=True)
    flat = f.reshape(len(t), 50, -1)
    gram = x.transpose(0, 2, 1) @ x
    ridge = (1e-6 + 1e-6*gram.trace(axis1=1, axis2=2))[:, None, None]
    coef = np.linalg.solve(gram + ridge*np.eye(3), x.transpose(0, 2, 1) @ flat)
    z = (flat - x @ coef).reshape(f.shape).astype('float32')
    rms = np.sqrt(np.mean(x*x, axis=1)+.0001)
    relative_strength = np.sqrt(np.mean(z*z, axis=1)+.0001) / (rms[:, :, None]+.06)
    context = np.column_stack([np.log1p(relative_strength).reshape(len(t), -1),
                               np.log1p(3*rms), np.tanh(3*q.mean(1)),
                               np.tanh(3*np.abs(x).mean(1)), np.exp(-t/400)])
    return z, context.astype('float32')


def training_record(base, nets, t, y):
    z, c = inputs(base, nets, t)
    p = base[t].astype(float)
    yc = y[t] - y[t].mean((0, 1))
    pc = p-p.mean((0, 1))
    scale = ((yc*pc).mean((0, 1))/(pc*pc).mean((0, 1)).clip(1e-7)).clip(.3, 3.)
    target = yc/scale
    target -= target.mean(1, keepdims=True)
    return z, c, (target-(p-p.mean(1, keepdims=True))).astype('float32')


class Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('weight', torch.zeros(3, N))
        self.register_buffer('center', torch.zeros(CONTEXT))
        self.register_buffer('scale', torch.ones(CONTEXT))
        self.gate = nn.Sequential(nn.Linear(CONTEXT, 16), nn.Tanh(), nn.Linear(16, 3*N))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def weights(self, c):
        return .20*torch.tanh(self.gate(((c-self.center)/self.scale).clamp(-4, 4))).reshape(-1, 3, N)

    def forward(self, z, c, dynamic):
        w = self.weight + (self.weights(c) if dynamic else 0)
        return (z*w.reshape(-1, 1, 3, N)).sum(-1)


def fit(rows):
    torch.manual_seed(2203)
    net = Mixer()
    z, c, y = [torch.from_numpy(np.concatenate(a)) for a in zip(*rows)]
    net.center.copy_(c.mean(0))
    net.scale.copy_(c.std(0).clamp_min(.05))
    with torch.no_grad():
        for k in range(3):
            x = z[:, :, k].reshape(-1, N).double()
            target = y[:, :, k].reshape(-1).double()
            g = x.T@x / len(x)
            reg = .65*torch.diag(g.diag().clamp_min(.0001))
            net.weight[k].copy_(torch.linalg.solve(g+reg, x.T@target/len(x)).clamp(-.30, .30))
    opt = torch.optim.AdamW(net.parameters(), lr=.002, weight_decay=.01)
    rng = np.random.default_rng(2203)
    for _ in range(180):
        ix = rng.integers(len(z), size=64)
        w = net.weights(c[ix])
        pred = (z[ix]*(net.weight+w)[:, None]).sum(-1)
        loss = ((pred-y[ix])**2).mean() + .005*w.square().mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval()


@torch.inference_mode()
def revised(net, p, forecasts, t, dynamic):
    z,c = inputs(p, forecasts, t)
    return p[t] + .35*net(torch.from_numpy(z), torch.from_numpy(c), dynamic).numpy()


def load():
    gls = rolling.gls
    parent = gls.incumbent.load()
    experts = [fn for fn, _ in rolling.panel.models(16)] + [gls.family.loader(m,16) for m in rolling.EXTRA]
    net = Mixer().eval()
    net.load_state_dict(torch.load(OUT/'net.pth', map_location='cpu', weights_only=True))
    mask = np.load(OUT/'mask.npy')
    def apply(raw, prev):
        p = parent(raw, prev)
        fs = [fn(raw) for fn in experts]
        q = p.copy()
        for start in range(0,len(p),96):
            t = np.arange(start,min(start+96,len(p)))
            q[t] = np.where(mask, revised(net, p, fs, t, True), p[t])
        return q
    return apply


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    history, records, folds, per_day = [], [], [], []
    for end in (8, 10, 12, 14):
        coeff = rolling.fit(history)
        model = fit(records) if records else None
        block = [[[], []], [[], []]]
        for d in (end, end+1):
            fs = rolling.sources(d)
            fixed = np.load(rolling.panel.OUT / f'panel_{d}.npy', mmap_mode='r')
            t = np.arange(len(fixed))
            adaptive = rolling.predict(fs, t, coeff)
            y = labels(d).reshape(-1, 50, 3)
            history.append(rolling.moments(fs, y, t[d%5::5]))
            for j,p in enumerate((adaptive,fixed)):
                records.append(training_record(p, fs[1:], t[d%19::19], y))
                if model is None:
                    continue
                ix = t[d%4::4]
                for k, dynamic in enumerate((False,True)):
                    q = revised(model, p, fs[1:], ix, dynamic)
                    row = tuple(a.reshape(-1,3) for a in (p[ix],q,y[ix]))
                    block[j][k].append(row)
                    if j==0 and dynamic:
                        sc = measure([row])
                        per_day.append(np.subtract(sc['candidate'],sc['base']))
                        if d==end:
                            tt=ix[ix<181]
                            short=revised(model,p[:181],[f[:181] for f in fs[1:]],tt,True)
                            np.testing.assert_allclose(short,q[:len(tt)],rtol=0,atol=2e-6)
        if model is not None:
            folds.append([[measure(rs) for rs in group] for group in block])
            print('days',end+1,end+2,[(s['baseline_score'],s['score']) for s in folds[-1][0]],flush=True)
    gains=np.array([[[np.subtract(s['candidate'],s['base']) for s in group] for group in f] for f in folds])
    keep=((gains[:,0,1].min(0)>0)&(gains[:,1,1].min(0)>0)&
          ((gains[:,0,1]-gains[:,0,0]).mean(0)>0)&((np.stack(per_day)>0).sum(0)>=4))
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,retained=keep.tolist()),indent=2))
    print('retained',keep,flush=True)
    if keep.any():
        np.save(OUT/'mask.npy',keep)
        torch.save(fit(records).state_dict(),OUT/'net.pth')
        walk=rolling.gls.best.walk
        res=walk.replay(load())
        (OUT/'raw_replay.json').write_text(json.dumps(res,indent=2))
        old=json.loads((rolling.gls.incumbent.OUT/'reload_verification.json').read_text())['pooled']['score']
        print('official',res['pooled']['score'],res['pooled']['score']-old,flush=True)
        if res['pooled']['score']>old:
            check=walk.replay(load())
            np.testing.assert_allclose(check['pooled']['candidate'],res['pooled']['candidate'],rtol=0,atol=1e-10)
            (OUT/'reload_verification.json').write_text(json.dumps(check,indent=2))


if __name__=='__main__':
    main()
