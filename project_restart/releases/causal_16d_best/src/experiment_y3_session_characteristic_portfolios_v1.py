"""Forecast same-clock long-short portfolio returns conditional on visible market and book states."""
import argparse
import json

import numpy as np

import experiment_y3_session_continuation_ridge_v1 as clock
from y3_marginal_replay_v1 import evaluate

parent = clock.parent
OUT = clock.OUT.parent / 'y3_session_characteristic_portfolios_v1'
ARMS = ('incumbent', 'reassigned stocks', 'unconditional premium', 'conditional clock premium',
        'directional sorts', 'activity and disagreement sorts', 'conditional above unconditional')
BUCKET = 384


def weights(raw, p):
    price = np.tanh(raw[:, :, 0].astype(float) - .00136)
    flow = np.tanh(.8*np.arcsinh(raw[:, :, [26,38,41,42]].astype(float)))
    flow = ((flow[:, :, 0]-flow[:, :, 1]) + (flow[:, :, 2]-flow[:, :, 3]))*.5
    recent = clock.rolling(price, 8)
    slow = clock.rolling(price, 96)
    pressure = clock.rolling(flow, 32)
    activity = clock.rolling(abs(price), 96)
    disagreement = np.abs(p[:, :, 1] - p[:, :, 2])
    chars = np.stack((recent-slow, pressure, slow, activity, disagreement), axis=2)
    chars -= chars.mean(1, keepdims=True)
    chars = np.tanh(chars / np.sqrt((chars**2).mean(1, keepdims=True)+.0001))
    chars -= chars.mean(1, keepdims=True)
    # Remove exposure to the incumbent cross-sectional Y3 at the same tick.
    level = p[:, :, 2:3].astype(float)
    level -= level.mean(1, keepdims=True)
    chars -= level * (level*chars).sum(1, keepdims=True)/(level**2).sum(1, keepdims=True).clip(1e-12)
    return chars.astype(np.float32)


def features(raw, p):
    f = weights(raw, p)
    # Portfolio returns on the short-horizon forecasts and their recent changes.
    near = np.einsum('tsf,tsh->tfh', f, p[:, :, :2])/50
    near = near.reshape(len(f), -1)
    x = np.column_stack((clock.design(raw, p), near, near-clock.rolling(near, 32)))
    return f, x


def record(raw, p, y, day):
    a = parent.exposure.bands(p)
    b = parent.exposure.bands(y)
    aa = a-a.mean((0,1))
    bb = b-b.mean((0,1))
    slope = ((aa*bb).sum((0,1))/(aa*aa).sum((0,1)).clip(.001)).clip(.45,2.5)
    err = (b/slope-a) @ np.array([.62,.38])
    records = []
    for lo, hi in parent.base.segments(len(p)):
        f, x = features(raw[lo:hi], p[lo:hi])
        # Ridge-stabilized cross-sectional projection of the settled excess returns on current characteristics.
        factor = f.transpose(0,2,1)
        premia = np.linalg.solve(factor@f/50 + .30*np.eye(f.shape[-1]),
                                  ((factor@err[lo:hi,:,None])/50))[:,:,0]
        grams, crosses = [], []
        for start in range(0, len(x), BUCKET):
            t = np.arange(max(0,start-BUCKET)+day%5, min(len(x)-60, start+2*BUCKET), 5)
            z = x[t]
            grams.append(z.T@z/max(len(t),1))
            crosses.append(z.T@premia[t]/max(len(t),1))
        records.append((np.stack(grams), np.stack(crosses)))
    return records


def fit(rows):
    state = {}
    for side, observations in enumerate(zip(*rows)):
        n = min(len(g) for g, c in observations)
        gs, cs = [np.stack([d[:n] for d in col]) for col in zip(*observations)]
        g, c = gs.mean(0), cs.mean(0)
        reliability = abs(c)/(abs(c) + cs.std(0)/np.sqrt(len(cs)) + .00001)
        reg = 2*np.eye(g.shape[-1])*g.diagonal(axis1=-2, axis2=-1).clip(.00015)[:,None,:]
        state[f'w{side}'] = np.linalg.solve(g+reg, c*reliability)
        state[f'm{side}'] = (c*reliability)[:,0]
    return state


def corrections(raw, p, state):
    out = []
    for side, (lo,hi) in enumerate(parent.base.segments(len(p))):
        f, x = features(raw[lo:hi], p[lo:hi])
        w, avg = state[f'w{side}'], state[f'm{side}']
        ids = (np.arange(len(x))//BUCKET).clip(max=len(w)-1)
        fitted = np.einsum('tc,tck->tk',x,w[ids])
        part = f*fitted[:,None,:]
        pooled = part.sum(2)
        constant = (f*avg[ids,None,:]).sum(2)
        out.append(np.stack((np.roll(pooled,13,axis=1), constant, pooled,
                             part[:,:,:3].sum(2), part[:,:,3:].sum(2), pooled-constant), axis=2))
    return (.33*np.concatenate(out)).astype(np.float32)


def train():
    hist, clockhist, obs, days = [], [], [], []
    for d in range(4,16):
        raw, refs, y = parent.history.training_day(d)
        refs = np.array(refs, copy=True)
        today = [parent.exposure.observations(p,y) for p in refs]
        if d>=8:
            if d%2==0:
                state, common, exposure = fit(hist), clock.params(clockhist), parent.exposure.fit(obs)[6]
            refs[:,:,:,2] += parent.apply_market(*parent.historical_heads(d),exposure)
            strict = np.load(parent.ART/'cutoff_reference_panel_v1'/f'panel_{d}.npy')
            group = []
            for i,p in enumerate([*refs,strict]):
                q = p.copy()
                q[:,:,2] += clock.signals(raw,p,common)[:,:,clock.CHOSEN-1]
                z = corrections(raw,p,state)
                if d==8 and i==0:
                    n = 2499
                    np.testing.assert_allclose(z[:n], corrections(raw[:n],p[:n],state), atol=4e-7,rtol=0)
                    assert abs(z.mean(1)).max()<2e-7
                group.append(parent.base.records(q,y,z))
            days.append(group)
        obs.append(tuple(np.mean(v,0) for v in zip(*today)))
        for data, records in ((hist,[record(raw,p,y,d) for p in refs]),
                              (clockhist,[clock.record(raw,p,y) for p in refs])):
            data.append([tuple(np.mean(v,0) for v in zip(*s)) for s in zip(*records)])
        print('completed training-period day', d+1, flush=True)
    def score(group):
        return np.array([parent.base.score(m,np.ones(len(ARMS)-1)) for m in np.sum(group,axis=0)])
    h = np.stack([score(days[:4]),score(days[4:])])
    daily = np.stack([score([d]) for d in days])
    pick = max(range(3,len(ARMS)), key=lambda j:h[0,:,:,j].min())
    ok = (h[:,:,:,pick].min()>0 and (h[:,:,0,pick]>h[:,:,0,1]).all()
          and (daily[:,:,0,pick]>0).sum(0).min()>=5)
    OUT.mkdir(exist_ok=True)
    report = dict(arms=ARMS, first_window_choice=pick, selected=pick if ok else 0,
                  windows=[[9,12],[13,16]], halves=h.tolist(), daily=daily.tolist())
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    if ok:
        np.savez(OUT/'model.npz',pick=pick,**fit(hist))
    print('selected', report['selected'], 'strict Y3',h[:,-1,0].round(6).tolist(),flush=True)


def correction():
    with np.load(OUT/'model.npz') as f:
        state = dict(f)
    common = clock.correction()
    return lambda raw,p: common(raw,p) + corrections(raw,p,state)[:,:,int(state['pick'])-1]


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('train','screen','verify'),nargs='?',default='train')
    mode=parser.parse_args().mode
    if mode=='train':
        train()
    else:
        evaluate(parent,OUT,correction,mode=='screen')
