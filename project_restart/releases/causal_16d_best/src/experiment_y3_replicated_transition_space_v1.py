"""Estimate reproducible stock differences in book dynamics from two disjoint sets of dates."""
import json

import numpy as np
import torch

import experiment_y3_stock_pressure_dynamics_v1 as incumbent
import experiment_y3_residual_peer_curve_v1 as audit

clock,curve=incumbent.clock,incumbent.curve
OUT=clock.ART/'y3_replicated_transition_space_v1'


def stock_modes(history):
    g=np.mean([s[1] for s in history],0)
    typical=g.mean(0)
    eigen,axes=np.linalg.eigh(typical)
    root=(axes*np.sqrt(eigen.clip(.00005)))@axes.T
    scale=np.sqrt(typical.diagonal()[:8].clip(.0001))
    splits=[]
    for days in (history[::2],history[1::2]):
        _,_,systems=incumbent.systems(days)
        matrices=systems[2]-systems[0]
        contrasts=(root@matrices)/scale
        contrasts=contrasts.reshape(50,-1)
        splits.append(contrasts-contrasts.mean(0))
    a,b=splits
    signal=(a@b.T+b@a.T)/2
    noise=(a-b)@(a-b).T/4
    values,directions=np.linalg.eigh(signal)
    uncertainty=np.einsum('ij,ij->j',directions,noise@directions)
    strength=values.clip(0)/(values.clip(0)+uncertainty+.00001)
    smoother=(directions*strength)@directions.T
    cosine=a@b.T/(np.linalg.norm(a,axis=1)[:,None]*np.linalg.norm(b,axis=1)[None,:]).clip(1e-10)
    return smoother,cosine,g


def geometry(history):
    mu,w,_=incumbent.fit(history)
    smoother,cosine,g=stock_modes(history)
    order=np.argsort(g.diagonal(axis1=1,axis2=2).sum(1))
    other=np.empty(50,dtype=int);other[order]=order.reshape(5,10)[:,::-1].ravel()
    shuffled=np.empty(50,dtype=int);shuffled[order]=np.roll(order.reshape(5,10),3,axis=1).ravel()
    far=w[2,:,:,1]
    np.testing.assert_allclose(far[other],w[1,:,:,1],atol=1e-7,rtol=0)
    common=far.mean(0)
    stable=smoother@(far-common)
    placebo=smoother[np.ix_(shuffled,shuffled)]@(far-common)
    bank=np.stack((far-far[other],stable-stable[other],placebo-placebo[other]),2).astype('float32')
    diagnostic=dict(effective_modes=float(smoother.trace()),identity_retrieval=int((cosine.argmax(1)==np.arange(50)).sum()),
                    self_cosine=float(cosine.diagonal().mean()),wrong_cosine=float(cosine[np.arange(50),other].mean()))
    return mu,bank,diagnostic


def predictions(v,mu,bank):
    f=(v-mu).transpose(1,0,2)@bank
    original,stable,placebo=np.moveaxis(clock.center(f.transpose(1,0,2)),2,0)
    return .17*original,[.17*(stable-original),.12*stable,.12*(stable-placebo)]


def run():
    history=[];folds=[];daily=[];repeat=[];blocks=[]
    for d in range(16):
        raw=clock.book.rawday(d);y=clock.labels(d).reshape(-1,50,3)
        if d>=8:
            if d%2==0:
                mu,bank,diagnostic=geometry(history)
                groups=[[],[]];repeat.append(diagnostic)
            old,z=predictions(incumbent.state(raw)[1],mu,bank)
            if d%2==0:
                _,short=predictions(incumbent.state(raw[:417])[1],mu,bank)
                for a,b in zip(z,short):np.testing.assert_allclose(a[:417],b,atol=3e-6,rtol=0)
            for j,ref in enumerate(curve.REFS):
                p=ref(d).reshape(y.shape).copy();p[:,:,2]+=old
                row=audit.blockstats(p,y,z);groups[j].append(row.sum(0))
                if not j:daily.append(audit.pooled(row.sum(0)).tolist());blocks.append(row)
            if d%2:
                folds.append([audit.pooled(np.sum(h,0)).tolist() for h in groups])
                print('ending',d+1,repeat[-1],folds[-1][0],flush=True)
        history.append(incumbent.stats(raw,y,d))
    a=np.array(folds)
    keep=((a>0).sum(0).min(0)>=3)&(a.mean(0).min(0)>0)&(a[-1].min(0)>0)&((np.array(daily)>0).sum(0)>=5)
    choices=np.flatnonzero(keep)
    pick=int(max(choices,key=lambda k:a[:,0,k].mean())) if len(choices) else None
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,folds=folds,daily=daily,replication=repeat,
        bootstrap=audit.intervals(blocks),arms=['replace with stable','additional stable','stable versus scrambled']),indent=2))
    if pick is not None:
        mu,bank,_=geometry(history);np.savez(OUT/'model.npz',mu=mu,bank=bank,pick=pick)
    return pick is not None


def correction():
    with np.load(OUT/'model.npz') as s:
        mu,bank,pick=s['mu'],s['bank'],int(s['pick'])
    return lambda raw: predictions(incumbent.state(raw)[1],mu,bank)[1][pick]


def layer():
    parent=incumbent.layer()
    delta=correction()
    def apply(raw,p):
        q=parent(raw,p)
        q[:,:,2]+=delta(raw)
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    parent,update=ancestor.load(),layer()
    return lambda raw,p:update(raw,parent(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if run():incumbent.impact.source.replay.verify(OUT,layer,incumbent.OUT)
