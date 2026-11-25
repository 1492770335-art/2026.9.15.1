"""Estimate the interaction of book dynamics and payoff on different dates, then test its incremental score."""
import json

import numpy as np
import torch

import experiment_y3_replicated_transition_space_v1 as traits
import experiment_y3_transition_payoff_identity_v1 as price
import experiment_y3_residual_peer_curve_v1 as audit

incumbent=traits.incumbent
clock,curve=incumbent.clock,incumbent.curve
OUT=clock.ART/'y3_crossfit_transition_interaction_v1'


def geometry(hist):
    mu,first,_=traits.geometry(hist)
    _,second,_=price.geometry(hist)
    pooled=np.mean([s[1] for s in hist],0)
    order=np.argsort(pooled.diagonal(axis1=1,axis2=2).sum(1))
    wrong=np.empty(50,dtype=int);wrong[order]=order.reshape(5,10)[:,::-1].ravel()
    halves=[hist[::2],hist[1::2]]
    paths=[price.projected.operators(h)[2][2,:,1] for h in halves]
    loads=[price.exposure(h)[0] for h in halves]
    gram=pooled.mean(0)
    agreement=((loads[0]@gram)*loads[1]).sum(1)
    energy=sum(((w@gram)*w).sum(1) for w in loads)*.5
    shrink=((agreement+.3*agreement.mean())/(energy+.3*energy.mean()+.000002)).clip(0,1)
    loads=[w*shrink[:,None] for w in loads]
    def combine(d,w):
        return 8*np.einsum('sij,sj->si',d-d[wrong],w-w[wrong])
    crossed=.5*(combine(paths[0],loads[1])+combine(paths[1],loads[0]))
    same=.5*(combine(paths[0],loads[0])+combine(paths[1],loads[1]))
    # Columns include both existing branches so that the merge is tested against each one separately.
    bank=np.stack((.17*first[:,:,0],.12*(first[:,:,1]-first[:,:,2]),
                   .17*second[:,:,3],.17*crossed,.17*same),axis=2).astype('float32')
    return mu,bank


def terms(v,mu,bank):
    q=((v-mu).transpose(1,0,2)@bank).transpose(1,0,2)
    return list(np.moveaxis(clock.center(q),2,0))


def candidates(rows):
    _,stable,interaction,crossed,same=rows
    return [stable,interaction,stable+interaction,stable+crossed,stable+interaction+.4*(crossed-same)]


def run():
    hist=[];folds=[];daily=[];blocks=[]
    for d in range(16):
        raw=clock.book.rawday(d);y=clock.labels(d).reshape(-1,50,3)
        if d>=8:
            if d%2==0:mu,bank=geometry(hist);groups=[[],[]]
            result=terms(incumbent.state(raw)[1],mu,bank)
            if d%2==0:
                short=terms(incumbent.state(raw[:347])[1],mu,bank)
                for a,b in zip(short,result):np.testing.assert_allclose(a,b[:347],atol=3e-6,rtol=0)
            z=candidates(result)
            for j,ref in enumerate(curve.REFS):
                p=ref(d).reshape(y.shape).copy();p[:,:,2]+=result[0]
                row=audit.blockstats(p,y,z);groups[j].append(row.sum(0))
                if not j:daily.append(audit.pooled(row.sum(0)).tolist());blocks.append(row)
            if d%2:
                folds.append([audit.pooled(np.sum(h,0)).tolist() for h in groups]);print('ending',d+1,folds[-1][0],flush=True)
        hist.append(incumbent.stats(raw,y,d))
    a=np.array(folds);tail=a[:,:,2:]
    # Pool complementary signals only when the combined forecast also wins against the two component forecasts.
    over=tail-a[:,:,:2].max(axis=2,keepdims=True)
    daily=np.array(daily)
    day_over=daily[:,2:]-daily[:,:2].max(axis=1,keepdims=True)
    ok=((over>0).sum(0).min(0)>=3)&(over.mean(0).min(0)>0)&(over[-1].min(0)>0)&((day_over>0).sum(0)>=5)
    choices=np.flatnonzero(ok)
    pick=int(max(choices,key=lambda k:tail[:,0,k].mean())+2) if len(choices) else None
    report=dict(selected=pick,folds=folds,daily=daily.tolist(),bootstrap=audit.intervals(blocks),
                gains_over_best_component=over.tolist(),arms=['stable','interaction','sum','cross-fitted sum','covariance corrected sum'])
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    if pick is not None:
        mu,bank=geometry(hist);np.savez(OUT/'model.npz',mu=mu,bank=bank,pick=pick)
    return pick is not None


def correction():
    with np.load(OUT/'model.npz') as s:
        mu,bank,pick=s['mu'],s['bank'],int(s['pick'])
    return lambda raw: candidates(terms(incumbent.state(raw)[1],mu,bank))[pick]


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
