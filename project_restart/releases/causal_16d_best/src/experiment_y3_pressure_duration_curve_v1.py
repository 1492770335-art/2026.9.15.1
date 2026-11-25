"""Forecast future pressure occupancy using empirical state age instead of memoryless exit rates."""
import json
import numpy as np
import torch
import experiment_y3_hidden_pressure_states_v1 as markov

curve=markov.curve
clock=curve.clock
incumbent=curve.incumbent
OUT=clock.ART/'y3_pressure_duration_curve_v1'
AGE=14


def next_age(old,trans):
    diag=trans.diagonal()
    current=np.empty_like(old)
    exit_trans=trans-np.diag(diag)
    current[:,:,0]=old.sum(2)@exit_trans
    current[:,:,1:]=old[:,:,:-1]*diag[None,:,None]
    current[:,:,-1]+=old[:,:,-1]*diag
    return current


def filtered_age(lik,trans,p0):
    a=np.zeros((lik.shape[1],len(p0),AGE));a[:,:,0]=p0
    for i,v in enumerate(lik):
        if i:a=next_age(a,trans)
        a*=v[:,:,None]
        a/=a.sum((1,2),keepdims=True)
        yield a


def standardize(raw,m):
    return ((markov.observations(raw)-m[0])/m[1]).clip(-4,4)


def duration(end,m):
    trans=m[-2];exit_trans=trans-np.diag(trans.diagonal())
    total=np.zeros((2,len(trans),AGE))
    for day in range(end):
        z=standardize(clock.book.rawday(day),m)[4+day%5::5]
        lik=markov.emissions(z,m[2],m[3])
        beta=np.ones_like(lik)
        for i in range(len(lik)-2,-1,-1):
            v=(lik[i+1]*beta[i+1])@trans.T
            beta[i]=v/v.mean(1,keepdims=True)
        for i,a in enumerate(filtered_age(lik[:-1],trans,m[-1])):
            v=lik[i+1]*beta[i+1]
            exit_prob=v@exit_trans.T
            stay=v*trans.diagonal()
            normal=(a.sum(2)*(exit_prob+stay)).sum(1)
            total[0]+=(a*(exit_prob/normal[:,None])[:,:,None]).sum(0)
            total[1]+=(a*((exit_prob+stay)/normal[:,None])[:,:,None]).sum(0)
    # Pool nearby ages and shrink log-odds toward the state's marginal exit probability.
    bins=np.digitize(np.arange(AGE),(1,2,3,5,8,12))
    rates=np.zeros_like(total[0]);prior=1-trans.diagonal()
    for b in np.unique(bins):
        sel=bins==b
        n=total[1][:,sel].sum(1);e=total[0][:,sel].sum(1)
        rates[:,sel]=((e+160*prior)/(n+160))[:,None]
    return rates.clip(.01,.95)


def propagation(trans,hazard):
    k=len(trans)
    exit_trans=trans-np.diag(trans.diagonal())
    exit_trans=exit_trans/exit_trans.sum(1,keepdims=True)
    p=np.zeros((k,AGE,k,AGE))
    for a in range(AGE):
        p[:,a,:,0]=hazard[:,a,None]*exit_trans
        p[np.arange(k),a,np.arange(k),min(AGE-1,a+1)]+=1-hazard[:,a]
    flat=p.reshape(k*AGE,k*AGE)
    current=np.eye(k).repeat(AGE,0);at=[]
    for h in range(12):
        if h>=4:at.append(current)
        current=flat@current
    return np.stack((np.mean(at[:4],0),np.mean(at[4:],0)),axis=2).reshape(k*AGE,-1)


def occupancy(raw,t,m,h):
    z=standardize(raw,m);lik=markov.emissions(z,m[2],m[3])
    static=propagation(m[-2],np.broadcast_to(1-m[-2].diagonal()[:,None],h.shape))
    moving=propagation(m[-2],h)
    table=np.column_stack((static,moving))
    nstates=m[-1].size
    latent=np.empty((len(t),50,table.shape[1]));instant=np.empty((len(t),50,nstates))
    destination=np.full(len(raw),-1);destination[t]=np.arange(len(t))
    for phase in range(min(5,len(raw))):
        for tick,a in zip(range(phase,len(raw),5),filtered_age(lik[phase::5],m[-2],m[-1])):
            if destination[tick]>=0:
                idx=destination[tick];instant[idx]=a.sum(2);latent[idx]=a.reshape(50,-1)@table
    return z[t],instant,np.split(latent,2,axis=2)


def features(raw,t,m,h):
    z,inst,fore=occupancy(raw,t,m,h)
    return [clock.center(np.concatenate((z,inst,f,
        (f[:,:,:,None]*z[:,:,None,:]).reshape(len(t),50,-1)),2)).astype('float32') for f in fore]


def train(end):
    m=markov.fit_states(end);h=duration(end,m)
    hist=[[],[]]
    for d in range(end):
        raw=clock.book.rawday(d);t=np.arange(d%7,len(raw)-61,7)
        y=clock.labels(d).reshape(-1,50,3);b,_=curve.target(y,t)
        outcome=np.stack((b[:,:,1:3].sum(2),b.sum(2),clock.center(y[t,:,2])),2).clip(-4,4).astype('float32')
        for v,x in zip(hist,features(raw,t,m,h)):v.append(curve.gram(x,outcome))
    return m,h,[curve.solve(*[np.mean(v,0) for v in zip(*q)],.45) for q in hist]


def predict(raw,t,m,h,w):
    old,new=[x@v for x,v in zip(features(raw,t,m,h),w)]
    return .20*(new-old)


def run():
    folds=[];days=[];hazards={}
    for end in (8,10,12,14):
        m,h,w=train(end);hazards[end]=h.tolist()
        groups=[[[],[],[]],[[],[],[]]]
        for day in (end,end+1):
            raw=clock.book.rawday(day);t=np.arange(day%4,len(raw),4)
            z=predict(raw,t,m,h,w)
            if day==end:
                sub=t[t<419]
                np.testing.assert_allclose(predict(raw[:419],sub,m,h,w),z[:len(sub)],rtol=0,atol=1e-6)
            y=clock.labels(day).reshape(-1,50,3)[t]
            for j,ref in enumerate(curve.REFS):
                p=ref(day).reshape(-1,50,3)[t]
                for k in range(3):groups[j][k].append((p,y,z[:,:,k]))
            days.append([curve.metric.gain(g[-1:]) for g in groups[0]])
        folds.append([[curve.metric.gain(g) for g in side] for side in groups]);print(end+1,folds[-1][0],flush=True)
    good=[a for a in range(3) if all(sum(f[j][a]['y3']>0 for f in folds)>=3 and np.mean([f[j][a]['y3'] for f in folds])>0
         and folds[-1][j][a]['y3']>0 for j in range(2)) and sum(d[a]['y3']>0 for d in days)>=5]
    pick=max(good,key=lambda a:np.mean([f[0][a]['y3'] for f in folds])) if good else None
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,folds=folds,daily=days,exit_rates=hazards),indent=2))
    if pick is not None:
        m,h,w=train(16);np.savez(OUT/'model.npz',**{f'm{i}':v for i,v in enumerate(m)},hazard=h,w=w,pick=pick)
    return pick is not None


def layer():
    previous=incumbent.layer()
    with np.load(OUT/'model.npz') as s:m=[s[f'm{i}'] for i in range(6)];h,w,pick=s['hazard'],s['w'],int(s['pick'])
    def apply(raw,p):
        q=previous(raw,p)
        q[:,:,2]+=predict(raw,np.arange(len(q)),m,h,w)[:,:,pick]
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    previous,act=ancestor.load(),layer()
    return lambda raw,p:act(raw,previous(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if run():incumbent.incumbent.base.chain.replay.verify(OUT,layer,incumbent.OUT)
