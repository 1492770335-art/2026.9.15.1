"""Factor the stock identity of a future book path and the identity of its price response separately."""
import json

import numpy as np
import torch

import experiment_y3_stock_pressure_dynamics_v1 as incumbent
import experiment_y3_projected_state_payoff_v1 as projected
import experiment_y3_residual_peer_curve_v1 as audit

clock,curve=incumbent.clock,incumbent.curve
OUT=clock.ART/'y3_transition_payoff_identity_v1'


def exposure(history):
    _,g,_,p=[np.mean(v,0) for v in zip(*history)]
    average=g.mean(0)
    prior=curve.solve(average,p.mean(0),.22)
    penalty=.7*np.diag(average.diagonal().clip(.001))+.00025*np.eye(len(prior))
    departure=np.linalg.solve(g+penalty,(p-g@prior)[:,:,None])[:,:,0]
    # Strip the stock-independent correction so only cross-stock variation remains.
    return departure-departure.mean(0),prior


def geometry(history):
    mu,price,ops=projected.operators(history)
    g=np.mean([s[1] for s in history],0)
    avg=g.mean(0)
    a,b=[exposure(history[k::2])[0] for k in (0,1)]
    match=((a@avg)*b).sum(1)
    energy=(((a@avg)*a).sum(1)+((b@avg)*b).sum(1))*.5
    mean_energy=float(energy.mean())
    shrink=((match+.3*match.mean())/(energy+.3*mean_energy+.000002)).clip(0,1)
    shift,_=exposure(history)
    shift=shift*shrink[:,None]
    order=np.argsort(g.diagonal(axis1=1,axis2=2).sum(1))
    wrong=np.empty(50,dtype=int);wrong[order]=order.reshape(5,10)[:,::-1].ravel()
    E=ops[2,:,1];F=ops[1,:,1]
    # Each of the following predicts a sum over eight successive five-tick intervals.
    def value(op,read):
        return np.einsum('sij,sj->si',op,read)
    c=shift[wrong]
    baseline=8*(E-F)@price
    identity=8*value(E,shift-c)
    timing=8*value(E-F,shift)
    interaction=8*value(E-F,shift-c)
    joint=8*(value(E,shift)-value(F,c))
    bank=np.stack((baseline,identity,timing,interaction,joint),2).astype('float32')
    diag=dict(replicating_names=int((match>0).sum()),adjustment_fraction=float(shrink.mean()),
              average_cross_moment=float(match.mean()),average_energy=mean_energy)
    return mu,bank,diag


def predictions(v,mu,bank):
    z=((v-mu).transpose(1,0,2)@bank).transpose(1,0,2)
    return [.17*a for a in np.moveaxis(clock.center(z),2,0)]


def run():
    hist=[];folds=[];daily=[];quality=[];blocks=[]
    for d in range(16):
        raw=clock.book.rawday(d);y=clock.labels(d).reshape(-1,50,3)
        if d>=8:
            if d%2==0:
                mu,w,q=geometry(hist);quality.append(q);groups=[[],[]]
            original,*z=predictions(incumbent.state(raw)[1],mu,w)
            if d%2==0:
                short=predictions(incumbent.state(raw[:371])[1],mu,w)
                for a,b in zip([original,*z],short):np.testing.assert_allclose(a[:371],b,atol=3e-6,rtol=0)
                prior=incumbent.fit(hist)
                expected=incumbent.changes(incumbent.forecast(incumbent.state(raw[:371])[1],*prior[:2]))[2]
                np.testing.assert_allclose(short[0],expected,rtol=0,atol=3e-6)
            for j,ref in enumerate(curve.REFS):
                p=ref(d).reshape(y.shape).copy();p[:,:,2]+=original
                row=audit.blockstats(p,y,z);groups[j].append(row.sum(0))
                if not j:daily.append(audit.pooled(row.sum(0)).tolist());blocks.append(row)
            if d%2:
                folds.append([audit.pooled(np.sum(h,0)).tolist() for h in groups])
                print('ending',d+1,quality[-1],folds[-1][0],flush=True)
        hist.append(incumbent.stats(raw,y,d))
    a=np.array(folds)
    ok=((a>0).sum(0).min(0)>=3)&(a.mean(0).min(0)>0)&(a[-1].min(0)>0)&((np.array(daily)>0).sum(0)>=5)
    choices=np.flatnonzero(ok)
    pick=int(max(choices,key=lambda k:a[:,0,k].mean())) if len(choices) else None
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,folds=folds,daily=daily,replication=quality,
           bootstrap=audit.intervals(blocks),arms=['payoff identity','transition of own payoff','interaction','joint']),indent=2))
    if pick is not None:
        mu,w,_=geometry(hist)
        np.savez(OUT/'model.npz',mu=mu,w=w,pick=pick)
    return pick is not None


def correction():
    with np.load(OUT/'model.npz') as s:
        mu,w,pick=s['mu'],s['w'],int(s['pick'])
    return lambda raw: predictions(incumbent.state(raw)[1],mu,w)[pick+1]


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
