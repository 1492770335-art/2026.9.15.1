"""Learn a payoff specifically for the book state predictable 20–60 ticks ahead on previous dates."""
import json
import numpy as np
import torch

import experiment_y3_stock_pressure_dynamics_v1 as dyn
import experiment_y3_residual_peer_curve_v1 as audit

curve,clock,incumbent=dyn.curve,dyn.clock,dyn.incumbent
OUT=clock.ART/'y3_projected_state_payoff_v1'


def operators(history):
    mu,prior,bank=dyn.systems(history)
    groups=[]
    for matrices in bank:
        rays=[]
        for w in matrices:
            trans=dyn.companion(w);power=np.eye(len(w));path=[]
            for j in range(12):
                path.append(power)
                power=power@trans
            rays.append(np.stack((np.mean(path[6:10],0),np.mean(path[4:],0))))
        groups.append(rays)
    return mu.astype('float32'),prior,np.array(groups,dtype='float32')


def project(v,mu,ops):
    return np.transpose((v-mu).transpose(1,0,2)@ops,(1,0,2))


def history_moments(v,y,t,model):
    mu,prior,ops=model
    blocks=dyn.impact.blocks(y,t)
    moments=[]
    for j,(start,stop) in enumerate(((6,10),(4,12))):
        # Scale to a per-five-tick return, the same units as the short readout.
        target=clock.center(blocks[:,:,start:stop].mean(2)).clip(-1.7,1.7)
        # Fit the same payoff in the same coordinates for both mappings.
        predicted=clock.center(project(v[t],mu,ops[2,:,j]))
        moments.append(curve.gram(predicted,target.astype('float32')))
    return moments


def fitted_payoff(records,prior):
    weights=[]
    for channel in zip(*records):
        g,b=[np.stack(v) for v in zip(*channel)]
        mean=g.mean(0)
        residual=b[:,:,0]-g@prior
        rhs=residual.mean(0)
        # Estimating payoff on anticipated states is ill-conditioned at long lags.
        # Anchor directions with little predictable variation to the short-return price loading.
        reg=np.diag(1.05*mean.diagonal().clip(.00065))+.0006*np.eye(len(mean))
        change=np.linalg.solve(mean+reg,rhs)
        se=residual.std(0)/np.sqrt(len(records))
        weights.append(prior+change*abs(rhs)/(abs(rhs)+se+.000015))
    return np.array(weights)


def changes(v,model,weights):
    mu,prior,ops=model
    variants=[]
    for j,scale in enumerate((4,8)):
        shift=ops[2,:,j]-ops[1,:,j]
        aligned=project(v,mu,shift@weights[j,:,None])[:,:,0]*scale
        original=project(v,mu,shift@prior[:,None])[:,:,0]*scale
        variants.extend((.17*clock.center(aligned),.17*clock.center(aligned-original)))
    return variants


def run():
    history,records,folds,daily,coeffs=[],[],[],[],[]
    for d in range(16):
        raw=clock.book.rawday(d);y=clock.labels(d).reshape(-1,50,3)
        if d>=4:
            if d%2==0:
                model=operators(history)
                if d>=8:
                    w=fitted_payoff(records,model[1]);groups=[[],[]]
                    coeffs.append(np.linalg.norm(w-model[1],axis=1).tolist())
            v=dyn.state(raw)[1]
            if d>=8:
                z=changes(v,model,w)
                if d%2==0:
                    for full,sub in zip(z,changes(dyn.state(raw[:419])[1],model,w)):
                        np.testing.assert_allclose(full[:419],sub,rtol=0,atol=3e-6)
                for j,ref in enumerate(curve.REFS):
                    s=audit.blockstats(ref(d).reshape(y.shape),y,z).sum(0);groups[j].append(s)
                    if not j:daily.append(audit.pooled(s).tolist())
                if d%2:
                    folds.append([audit.pooled(np.sum(g,0)).tolist() for g in groups])
                    print('ending',d+1,folds[-1][0],flush=True)
            t=np.arange(d%5,len(raw)-60,5)
            records.append(history_moments(v,y,t,model))
        history.append(dyn.stats(raw,y,d))
    a=np.array(folds)
    keep=((a>0).sum(0).min(0)>=3)&(a.mean(0).min(0)>0)&(a[-1].min(0)>0)&((np.array(daily)>0).sum(0)>=5)
    choices=np.flatnonzero(keep)
    pick=int(max(choices,key=lambda k:a[:,0,k].mean())) if len(choices) else None
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,folds=folds,daily=daily,loading_change=coeffs,
       arms=['projected middle','middle versus short loading','projected far','far versus short loading']),indent=2))
    if pick is not None:
        mu,prior,ops=operators(history)
        np.savez(OUT/'model.npz',mu=mu,prior=prior,ops=ops,w=fitted_payoff(records,prior),pick=pick)
    return pick is not None


def layer():
    parent=incumbent.layer()
    with np.load(OUT/'model.npz') as s:
        model=(s['mu'],s['prior'],s['ops']);w=s['w'];pick=int(s['pick'])
    def apply(raw,p):
        q=parent(raw,p)
        q[:,:,2]+=changes(dyn.state(raw)[1],model,w)[pick]
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    parent,update=ancestor.load(),layer()
    return lambda raw,p:update(raw,parent(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if run():dyn.impact.source.replay.verify(OUT,layer,incumbent.OUT)
