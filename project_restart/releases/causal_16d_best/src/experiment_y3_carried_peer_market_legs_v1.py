"""Measure how much each spatial component of the improved 20-tick forecast carries into Y3."""
import json
import sys
import numpy as np
import torch

import joint_separated_peer_references as history
import experiment_y3_settled_factor_confidence_v1 as observed

parent,stack,trial=history.incumbent,history.stack,history.trial
peer=parent.peer
OUT=trial.clock.ART/'y3_carried_peer_market_legs_v1'
ARMS=('unchanged','full medium carry','half medium carry','market component','relative components',
      'market driven by other stocks plus market to stock','estimated total carry','estimated separate components',
      'date-shrunk separate components','permuted recipients','carry with residual horizon adjustment')


def legs(p,w,basis):
    x=peer.coordinates(p,basis)
    market=(x@w[:,0])[:,None]*basis[:,0]
    mr=(x[:,:3]@w[:3,1:])@basis[:,1:].T
    rr=(x[:,3:]@w[3:,1:])@basis[:,1:].T
    return np.stack((market,mr,rr),axis=2)


def record(p,q,y,w,basis):
    z=legs(p,w,basis)
    fake=np.roll(legs(np.roll(p,17,axis=1),w,basis),-17,axis=1)
    # Features have forecast units. Use true labels only in these training/evaluation statistics.
    a=np.concatenate((np.ones_like(y[:,:,:1]),q[:,:,1:],z,fake,y[:,:,1:]),axis=2).reshape(-1,11).astype(float)
    return a.T@a


def fit(records):
    days=np.mean(records,axis=1)
    c=stack.covariance(days.sum(0))
    b2,b3=[np.clip(c[j,-2+j]/c[j,j],.35,3) for j in (0,1)]
    co=observed.historical_geometry()[1][:,1]
    ratio=co[2]/co[1]*b2/b3
    idx=np.arange(2,5)
    g=c[np.ix_(idx,idx)]-np.outer(c[idx,1],c[1,idx])/c[1,1]
    def gradient(m):
        cv=stack.covariance(m)
        beta=np.clip(cv[1,-1]/cv[1,1],.35,3)
        return cv[idx,-1]/beta-cv[idx,1]
    v=gradient(days.sum(0))
    date=np.stack([gradient(a)/a[0,0] for a in days])
    reliability=abs(date.mean(0))/(abs(date.mean(0))+date.std(0)/np.sqrt(len(date))+1.e-9)
    penalty=np.diag(g.diagonal().clip(1.e-8))
    def solve(rhs):
        return (.6*np.linalg.solve(g+.9*penalty,rhs)).clip(-2,2)
    carry=ratio*np.ones(3)
    scale=.55*float(v.sum()/max(1.e-8,g.sum()))
    separate=solve(v)
    weights=[np.zeros(3),carry,.5*carry,carry*[1,0,0],carry*[0,1,1],carry*[1,1,0],
             np.clip(scale,-2,2)*np.ones(3),separate,solve(v*reliability),carry,
             carry+solve((v-g@carry)*reliability)]
    return np.array(weights)


def gains(m,weights):
    c=stack.covariance(m)
    a=[]
    for j,w in enumerate(weights):
        v=np.zeros(8)
        v[1]=1
        start=5 if j==9 else 2
        v[start:start+3]=w
        a.append(v@c[:-2,-1]/np.sqrt((v@c[:-2,:-2]@v)*c[-1,-1]))
    return np.array(a)-a[0]


def warmup():
    basis=peer.U()[0].astype(float)
    early=history.study.screen.history.market.prior.path.early
    records=[]
    for d in range(4,8):
        raw=trial.clock.book.rawday(d)
        p=early.load(d,raw)
        y=trial.clock.labels(d).reshape(-1,50,3)
        records.append([peer.moments(p,y,basis)])
    return records


def train():
    OUT.mkdir(exist_ok=True)
    basis=peer.U()[0].astype(float)
    history_rows=[]
    grams=[]
    folds,daily=[],[]
    for d in range(8,16):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        before=history.previous.references(d,raw)
        forecasts=history.references(d,raw)
        if d%2==0:
            w=parent.readouts(history_rows if d>=10 else warmup(),1)[6]
            group=[]
            if d>=10:
                heads=fit(grams)
        rows=[]
        for p,q in zip(before,forecasts):
            if d>=10:
                np.testing.assert_allclose(p[:,:,1]+legs(p,w,basis).sum(2),q[:,:,1],rtol=0,atol=1e-6)
            rows.append(record(p,q,y,w,basis))
        grams.append(rows)
        history_rows.append([peer.moments(p,y,basis) for p in before])
        if d>=10:
            group.append(rows)
            daily.append([gains(m,heads) for m in rows])
            if d%2:
                folds.append([gains(m,heads) for m in np.sum(group,0)])
    f,dd=np.array(folds),np.array(daily)
    avg=f.mean(0)
    valid=[j for j in (1,2,3,4,5,6,7,8,10) if avg[:,j].min()>0 and (avg[:,j]>avg[:,9]).all()
           and f[-1,:,j].min()>0 and (f[:,:,j]>0).sum(0).min()>=2 and (dd[:,:,j]>0).sum(0).min()>=4]
    chosen=max(valid,key=lambda j:avg[:,j].min()) if valid else 0
    print('Y3',ARMS[chosen],np.round(avg,6),flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(selected=chosen,arms=ARMS,folds=f.tolist(),daily=dd.tolist(),mean=avg.tolist())))
    if chosen:
        np.savez(OUT/'model.npz',coef=fit(grams)[chosen],selected=chosen)
    return bool(chosen)


def adjustment():
    with np.load(OUT/'model.npz') as f:
        coef=f['coef']
    with np.load(parent.OUT/'model.npz') as f:
        basis=f['basis'].astype(float)
        w=f['w'][:,:,1]
    def apply(old,p):
        q=p.copy()
        q[:,:,2]+=(legs(old,w,basis)@coef).astype('float32')
        return q
    return apply


def layer():
    before=history.previous.incumbent.layer()
    current=parent.adjustment()
    adjust=adjustment()
    def apply(raw,initial):
        old=before(raw,initial)
        return adjust(old,current(old))
    return apply


def replay_with():
    apply=adjustment()
    def read(d,raw):
        p,h,n=history.read(d,raw)
        old,head,_=history.previous.read(d,raw)
        return apply(old,p),apply(head,h),n
    return stack.verifier,read


if __name__=='__main__':
    torch.set_num_threads(2)
    if sys.argv[1:]==['replay'] or train():
        peer.replay.replay(sys.modules[__name__])
