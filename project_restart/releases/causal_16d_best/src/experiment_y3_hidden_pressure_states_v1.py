"""Identify persistent latent pressure states with a learned transition matrix and test their forward payoffs."""
import json
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
import experiment_y3_partial_forward_curve_v1 as curve
import experiment_y3_peer_arrival_alignment_v1 as arrival

incumbent=curve.incumbent
clock=curve.clock
OUT=clock.ART/'y3_hidden_pressure_states_v1'
K=5
STEP=5


def observations(raw):
    x=clock.center(arrival.signals(raw)[:,:,[0,1,3,4,5,7]])
    cs=np.concatenate((np.zeros_like(x[:1],dtype=float),x.cumsum(0,dtype=float)))
    end=np.arange(1,len(x)+1);left=(end-STEP).clip(0)
    return ((cs[end]-cs[left])/(end-left)[:,None,None]).astype('float32')


def emissions(x,mu,var):
    score=-.5*(((x[...,None,:]-mu)**2/var).sum(-1)+np.log(var).sum(1))
    return np.exp(score-score.max(-1,keepdims=True)).clip(1e-15)


def fit_states(end):
    seqs=[observations(clock.book.rawday(d))[4+d%5::STEP] for d in range(end)]
    means=np.mean([s.mean(0) for s in seqs],0)
    std=np.sqrt(np.mean([(s-means).var(0) for s in seqs],0)+.02)
    n=min(map(len,seqs))
    z=np.concatenate([((s[:n]-means)/std).clip(-4,4) for s in seqs],1).astype('float32')
    training=z[::5].reshape(-1,z.shape[-1])
    mu=MiniBatchKMeans(n_clusters=K,n_init=3,batch_size=2048,random_state=671).fit(training).cluster_centers_
    var=np.ones_like(mu);p0=np.ones(K)/K
    trans=.75*np.eye(K)+.25/K
    for _ in range(5):
        lik=emissions(z,mu,var)
        alpha=np.empty_like(lik);last=np.broadcast_to(p0,(z.shape[1],K))
        for i,obs in enumerate(lik):
            update=(last@trans)*obs
            last=update/update.sum(1,keepdims=True);alpha[i]=last
        counts=np.zeros((K,K));occupancy=np.zeros(K);summ=np.zeros_like(mu,dtype=float);sq=summ.copy()
        back=np.ones_like(alpha[0],dtype=float)
        for i in range(n-1,-1,-1):
            g=alpha[i]*back;g/=g.sum(1,keepdims=True)
            occupancy+=g.sum(0);summ+=g.T@z[i];sq+=g.T@(z[i]**2)
            if i:
                v=lik[i]*back
                prev=v@trans.T
                denom=(prev*alpha[i-1]).sum(1)
                counts+=trans*((alpha[i-1]/denom[:,None]).T@v)
                back=prev/prev.mean(1,keepdims=True)
            else:p0=(g.sum(0)+1)/(len(g)+K)
        trans=counts+8*np.eye(K)+1
        trans/=trans.sum(1,keepdims=True)
        mu=summ/occupancy[:,None];var=(sq/occupancy[:,None]-mu**2).clip(.07,12)
    prior=p0.copy()
    for _ in range(120):prior=prior@trans
    return means,std,mu,var,trans,prior


def traces(raw,model):
    means,std,mu,var,trans,prior=model
    x=((observations(raw)-means)/std).clip(-4,4)
    likelihood=emissions(x,mu,var)
    independent=likelihood*prior;independent/=independent.sum(2,keepdims=True)
    ordered=np.empty_like(independent);ema=np.empty_like(ordered)
    persist=float((prior*np.diag(trans)).sum())
    for phase in range(min(STEP,len(x))):
        last=sm=np.broadcast_to(prior,(50,K))
        for t in range(phase,len(x),STEP):
            v=(last@trans)*likelihood[t]
            last=v/v.sum(1,keepdims=True);ordered[t]=last
            sm=persist*sm+(1-persist)*independent[t];ema[t]=sm
    return x,independent,ema,ordered


def design(trace,t):
    x,instant,ema,ordered=trace
    common=[x[t],instant[t]]
    def columns(s):
        p=s[t]
        interaction=p[:,:,:,None]*x[t,:,None,:]
        return clock.center(np.concatenate((*common,p,interaction.reshape(len(t),50,-1)),2)).astype('float32')
    return [columns(s) for s in (ema,ordered)]


def fit(end):
    model=fit_states(end);hist=[[],[]]
    for d in range(end):
        raw=clock.book.rawday(d);t=np.arange(d%7,len(raw)-61,7)
        y=clock.labels(d).reshape(-1,50,3)
        band,_=curve.target(y,t)
        truth=np.stack((band[:,:,1:3].sum(2),band.sum(2),clock.center(y[t,:,2])),2).clip(-4,4)
        for h,f in zip(hist,design(traces(raw,model),t)):h.append(curve.gram(f,truth.astype('float32')))
    return model,[curve.solve(*[np.mean(v,0) for v in zip(*h)],.45) for h in hist]


def delta(raw,t,model,weights):
    trace=traces(raw,model)
    old,new=[f@w for f,w in zip(design(trace,t),weights)]
    return .20*(new-old)


def run():
    folds=[];days=[];geometry={}
    for end in (8,10,12,14):
        m,w=fit(end);groups=[[[],[],[]],[[],[],[]]]
        geometry[end]=dict(occupancy=m[-1].tolist(),expected_duration_ticks=(5/(1-m[-2].diagonal())).tolist())
        for day in (end,end+1):
            raw=clock.book.rawday(day);t=np.arange(day%4,len(raw),4)
            z=delta(raw,t,m,w)
            if day==end:
                sub=t[t<399];np.testing.assert_allclose(delta(raw[:399],sub,m,w),z[:len(sub)],rtol=0,atol=2e-6)
            y=clock.labels(day).reshape(-1,50,3)[t]
            for j,ref in enumerate(curve.REFS):
                p=ref(day).reshape(-1,50,3)[t]
                for k in range(3):groups[j][k].append((p,y,z[:,:,k]))
            days.append([curve.metric.gain(g[-1:]) for g in groups[0]])
        folds.append([[curve.metric.gain(g) for g in side] for side in groups]);print(end+1,folds[-1][0],flush=True)
    good=[a for a in range(3) if all(sum(f[j][a]['y3']>0 for f in folds)>=3 and np.mean([f[j][a]['y3'] for f in folds])>0
            and folds[-1][j][a]['y3']>0 for j in (0,1)) and sum(d[a]['y3']>0 for d in days)>=5]
    pick=max(good,key=lambda a:np.mean([f[0][a]['y3'] for f in folds])) if good else None
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,folds=folds,daily=days,state_geometry=geometry,
        arms=['middle','far','whole']),indent=2))
    if pick is not None:
        m,w=fit(16);np.savez(OUT/'model.npz',**{f'm{i}':v for i,v in enumerate(m)},w=w,pick=pick)
    return pick is not None


def layer():
    previous=incumbent.layer()
    with np.load(OUT/'model.npz') as s:m=[s[f'm{i}'] for i in range(6)];w=s['w'];pick=int(s['pick'])
    def apply(raw,p):
        q=previous(raw,p);trace=traces(raw,m)
        for t in np.array_split(np.arange(len(raw)),max(1,(len(raw)+399)//400)):
            old,new=[x@v[:,pick] for x,v in zip(design(trace,t),w)]
            q[t,:,2]+=.2*(new-old)
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    parent,act=ancestor.load(),layer()
    return lambda raw,p:act(raw,parent(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if run():incumbent.incumbent.base.chain.replay.verify(OUT,layer,incumbent.OUT)
