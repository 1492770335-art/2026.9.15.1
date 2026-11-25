"""Test whether independently observed overshoots in two portfolios identify reliable market corrections."""
import json
import sys
import numpy as np
import torch
import y3_market_flow_references as history

market=history.incumbent
trial=market.trial
OUT=trial.clock.ART/'y3_overshoot_basket_agreement_v1'
ARMS=('uniform attenuation','discount conflicting evidence','reward agreeing evidence','signed agreement adjustment',
      'other-time agreement','scale by common fraction','consensus of total corrections','uniform amplification')


def signal(p,z,m):
    j=int(m['pick'])
    if market.ARMS[j]!='overshooting moves':
        raise ValueError('Expected the jointly selected overshoot market model.')
    weights=m['w'][:,j]
    before=[np.column_stack((p[:,1-i::2].mean(1),x))@weights for i,x in enumerate(z)]
    # p already contains the market correction; solve its one-dimensional linear feedback exactly.
    common=.5*(before[0]+before[1])/(1+weights[2])
    a,b=[x-common*weights[2] for x in before]
    evidence=[x[:,18:24]@weights[21:27] for x in z]
    left,right=evidence
    def confidence(u,v):
        return 2*u*v/(u*u+v*v+1e-8)
    strength=confidence(left,right)
    wrong=np.zeros_like(right);wrong[32:]=right[:-32]
    shared=np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0.)
    return common,[-.3*common,.55*common*np.minimum(strength,0),.4*common*np.maximum(strength,0),
                   .4*common*strength,.4*common*confidence(left,wrong),
                   -.55*common*(1-abs(strength)),.5*(shared-common),.3*common]


def run():
    OUT.mkdir(exist_ok=True)
    saved=json.loads((market.OUT/'results.json').read_text())
    rows=[];folds=[];days=[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        z=market.pools(raw,market.prior.teacher())
        y=trial.clock.labels(d).reshape(-1,50,3)
        base=[market.prior.path.early.load(d,raw)] if d<8 else history.previous.references(d,raw)
        if d>=8:
            refs=history.references(d,raw)
            if d%2==0:
                params=dict(market.fit(rows),pick=saved['selected']);groups=[[],[]]
            with np.load(history.CACHE/f'day_{d:02d}.npz') as f:expected=f['signals']
            for j,p in enumerate(refs):
                current,changes=signal(p,z,params)
                np.testing.assert_allclose(current,expected[j],rtol=0,atol=1e-6)
                if j==0 and d==8:
                    s=signal(p[:311],market.pools(raw[:311],params['teacher']),params)[1]
                    for u,v in zip(changes,s):np.testing.assert_allclose(u[:311],v,rtol=0,atol=1e-6)
                delta=[np.broadcast_to(v[:,None],p.shape[:2]) for v in changes]
                bs=trial.scoring.audit.blockstats
                stats=[bs(p,y,delta).sum(0),bs(p.mean(1,keepdims=True),y.mean(1,keepdims=True),[v[:,:1] for v in delta]).sum(0)]
                groups[j].append(stats)
                if j==0:days.append(trial.scoring.audit.pooled(stats[0]))
            if d%2:folds.append([[trial.scoring.audit.pooled(g) for g in np.sum(c,axis=0)] for c in groups])
        records=[market.record(z,p,y) for p in base]
        rows.append([np.mean(v,axis=0) for v in zip(*records)])
    f=np.array(folds);d=np.array(days);good=[]
    for j in (1,2,3,5,6):
        v=f[:,:,0,j]
        if ((v>0).sum(0).min()>=3 and v.mean(0).min()>0 and v[-1].min()>0 and
            (d[:,j]>0).sum()>=5 and f[:,:,1,j].mean(0).min()>0 and
            all((v-f[:,:,0,k]).mean(0).min()>0 for k in (0,4,7))):
            good.append(j)
    picked=max(good,key=lambda j:f[:,0,0,j].mean()) if good else None
    (OUT/'results.json').write_text(json.dumps(dict(selected=picked,arms=ARMS,folds=f.tolist(),daily=d.tolist()),indent=2))
    print('selected',picked,flush=True)
    if picked is not None:
        np.savez(OUT/'model.npz',picked=picked)
        history.outer(sys.modules[__name__])


def correction():
    with np.load(OUT/'model.npz') as f:j=int(f['picked'])
    with np.load(market.OUT/'model.npz') as f:m=dict(f)
    return lambda raw,p:signal(p,market.pools(raw,m['teacher']),m)[1][j][:,None]


def layer():
    predict,update=market.layer(),correction()
    def apply(raw,p):
        q=predict(raw,p)
        q[:,:,2]+=update(raw,q)
        return q
    return apply


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
