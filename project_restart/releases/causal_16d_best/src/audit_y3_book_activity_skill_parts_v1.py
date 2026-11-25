"""Measure frozen effects of ordinary horizon states, peer repricing and receiver activity."""
import json
import numpy as np
import torch
import y3_book_activity_skill_references as refs

model = refs.incumbent
NAMES = ('forecast-only control','joint intercept and forecast state','peer activity contribution',
         'receiver activity','receiver activity times agreement','all contextual terms')


def moments(raw,p,y,st):
    z = refs.trial.clock.center(p.astype(float))
    l = model.activity(raw)
    parts = np.empty((*p.shape[:2],len(NAMES)),float)
    for side in (0,1):
        own,s,a,g = model.views(z,l,side)
        read = model.coeff_projection(own,st['risk'])
        b = st['liquid']
        control = read(s[:,:13]@st['base'][:,:3])+a*read(s[:,:13]@st['base'][:,3:6])
        base = read(s[:,:13]@b[:13,:3])
        total = read(s@b[:,:3])
        receiver = g*read(s@b[:,6:9])
        joint = a*g*read(s@b[:,9:])
        parts[:,1-side::2] = np.stack((control,base,total-base,receiver,joint,total+receiver+joint),axis=2)
    q = (p[:,:,2:].astype(float)+parts)
    v = np.concatenate((np.ones_like(q[:,:,:1]),p[:,:,2:],q,y[:,:,None]),axis=2)
    v = v.reshape(-1,v.shape[-1])
    return v.T@v


def gains(m):
    c = refs.stack.covariance(m)
    ic = c[:-1,-1]/np.sqrt(c.diagonal()[:-1]*c[-1,-1])
    return dict(zip(NAMES,ic[1:]-ic[0]))


def run():
    rows,folds = [],[]
    for d in range(8,16):
        raw = refs.trial.clock.book.rawday(d)
        y = refs.trial.clock.labels(d).reshape(-1,50,3)
        pp = refs.previous.references(d,raw)
        if d>=10:
            if d%2==0:
                st = model.fit(rows)
                group = []
            group.append([moments(raw,p,y[:,:,2],st) for p in pp])
            if d%2:
                folds.append([gains(m) for m in np.sum(group,axis=0)])
        rows.append(tuple(np.mean(v,axis=0) for v in zip(*[model.record(raw,p,y) for p in pp])))
    with np.load(model.OUT/'model.npz') as f:
        st = dict(f)
    held = []
    for d in range(16,20):
        raw = refs.trial.clock.book.rawday(d)
        y = refs.trial.clock.labels(d).reshape(-1,50,3)
        p = refs.previous.read(d,raw)[0]
        held.append(moments(raw,p,y[:,:,2],st))
    pooled = gains(sum(held))
    check = json.loads((model.OUT/'reload_verification.json').read_text())['pooled']
    assert abs(pooled[NAMES[-1]]-(check['candidate'][2]-check['base'][2]))<1e-8
    r = dict(rolling=folds,validation=pooled,daily=[gains(m) for m in held],
             scope='Effect of isolated frozen terms on Y3 correlation; these changes are not additive.')
    (model.OUT/'activity_attribution.json').write_text(json.dumps(r,indent=2))
    print(pooled,flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
