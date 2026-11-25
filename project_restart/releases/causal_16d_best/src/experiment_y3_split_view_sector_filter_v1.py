"""Estimate sector order flow using lagged observations from two disjoint stock panels."""
import json
import numpy as np
import torch
import experiment_y3_peer_arrival_alignment_v1 as arrival

incumbent=arrival.incumbent
clock=incumbent.clock
OUT=clock.ART/'y3_split_view_sector_filter_v1'


def panels(ids):
    rng=np.random.default_rng(14622)
    parts=np.empty(50,int)
    for group in np.unique(ids):
        columns=rng.permutation(np.flatnonzero(ids==group))
        parts[columns]=np.arange(len(columns))%3
    return parts


def observations(raw,ids,parts,extract=arrival.signals):
    own=extract(raw)
    means=np.stack([np.stack([own[:,(ids==g)&(parts==p)].mean(1) for g in range(4)],1)
                    for p in range(3)],1)
    # Each instrument is made market-relative using only stocks belonging to that instrument.
    means-=means.mean(2,keepdims=True)
    return own,means


def design(means,t,p):
    donors=(means[:,(p+1)%3]+means[:,(p+2)%3])*.5
    cs=np.concatenate((np.zeros_like(donors[:1],dtype=float),donors.cumsum(0,dtype=float)))
    blocks=[]
    for near,far in ((0,8),(8,32),(32,96)):
        right=(t-near).clip(0);left=(t-far).clip(0)
        v=(cs[right]-cs[left])/(right-left).clip(1)[:,None,None]
        blocks.append(v)
    # Use the same time bins and channels in both estimators. The control sees their
    # average but cannot identify which earlier interval contributed a specific signal.
    mean=sum(blocks)/3
    common=np.concatenate((donors[t],mean),2)
    return [common,np.concatenate((common,blocks[0]-mean,blocks[1]-mean),2)]


def moments(x,y):
    x=x.reshape(-1,x.shape[-1]).astype(float)
    y=y.reshape(len(x),-1)
    return x.T@x/len(x),x.T@y/len(x)


def solve(history,strength):
    g,b=[np.mean(v,0) for v in zip(*history)]
    penalty=strength*np.diag(g.diagonal().clip(.000025))+.000015*np.eye(len(g))
    return np.linalg.solve(g+penalty,b)


def fit_filters(end,ids,parts,extract=arrival.signals):
    hist=[[],[]]
    for d in range(end):
        _,views=observations(clock.book.rawday(d),ids,parts,extract)
        t=np.arange(96+d%5,len(views),5)
        for p in range(3):
            for h,x in zip(hist,design(views,t,p)):
                h.append(moments(x,views[t,p]))
    return [solve(h,.15) for h in hist]


def features(raw,t,ids,parts,filters,extract=arrival.signals):
    own,views=observations(raw,ids,parts,extract)
    now=clock.center(own[t])
    prev=clock.center(own[(t-16).clip(0)])
    latent=[np.empty_like(now) for _ in filters]
    mse=np.zeros((2,2))
    for p in range(3):
        cols=np.flatnonzero(parts==p)
        for k,(x,w) in enumerate(zip(design(views,t,p),filters)):
            estimate=x@w
            latent[k][:,cols]=estimate[:,ids[cols]]
            mse[k]+=[np.square(estimate-views[t,p]).sum(),np.square(views[t,p]).sum()]
    xs=[]
    for z in latent:
        signed=np.tanh(3*z)
        xs.append(clock.center(np.concatenate((now,now-prev,z,signed,
            signed*np.abs(now),signed*np.abs(now-prev),signed*np.tanh(2*now*z)),2)))
    return xs,mse


def train(end,extract=arrival.signals):
    ids=arrival.sector.grouping(end);parts=panels(ids)
    filters=fit_filters(end,ids,parts,extract);hs=[[],[]]
    for day in range(end):
        raw=clock.book.rawday(day);t=np.arange(day%7,len(raw)-60,7)
        y=clock.center(clock.labels(day).reshape(-1,50,3)[t])
        truth=np.stack((y[:,:,2]-clock.RATIO*y[:,:,1],y[:,:,2]),2).clip(-4,4)
        for h,x in zip(hs,features(raw,t,ids,parts,filters,extract)[0]):h.append(moments(x,truth))
    return ids,parts,filters,[solve(h,.35) for h in hs]


def prediction(raw,t,fitted,extract=arrival.signals):
    ids,parts,filters,w=fitted
    xx,mse=features(raw,t,ids,parts,filters,extract)
    return .2*(xx[1]@w[1]-xx[0]@w[0]),mse


def run():
    folds=[];daily=[];reconstruction=[]
    for end in (8,10,12,14):
        fit=train(end);groups=[[[],[]],[[],[]]];mse=0.
        for day in (end,end+1):
            raw=clock.book.rawday(day);t=np.arange(day%4,len(raw),4)
            z,m=prediction(raw,t,fit);mse+=m
            if day==end:
                sub=t[t<419]
                np.testing.assert_allclose(prediction(raw[:419],sub,fit)[0],z[:len(sub)],rtol=0,atol=2e-6)
            truth=clock.labels(day).reshape(-1,50,3)[t]
            for j,ref in enumerate(arrival.sector.incumbent.REFS):
                p=ref(day).reshape(-1,50,3)[t]
                for k in (0,1):groups[j][k].append((p,truth,z[:,:,k]))
            daily.append([incumbent.base.chain.metric.gain(g[-1:]) for g in groups[0]])
        r2=1-mse[:,0]/mse[:,1]
        reconstruction.append(r2.tolist())
        folds.append([[incumbent.base.chain.metric.gain(g) for g in side] for side in groups])
        print(end+1,'sensor R2',r2,'gains',folds[-1][0],flush=True)
    good=[k for k in (0,1) if all(sum(f[j][k]['y3']>0 for f in folds)>=3 and folds[-1][j][k]['y3']>0
         and np.mean([f[j][k]['y3'] for f in folds])>0 for j in (0,1)) and sum(d[k]['y3']>0 for d in daily)>=5
         and np.mean([r[1]-r[0] for r in reconstruction])>0]
    pick=max(good,key=lambda k:np.mean([f[0][k]['y3'] for f in folds])) if good else None
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,selected=pick,
        held_out_sensor_r2=reconstruction,arms=['far target','whole Y3 target']),indent=2))
    if pick is not None:
        ids,parts,filters,w=train(16)
        np.savez(OUT/'model.npz',ids=ids,parts=parts,a=filters[0],b=filters[1],w=np.stack(w),pick=pick)
    return pick is not None


def layer():
    prev=incumbent.layer()
    with np.load(OUT/'model.npz') as s:
        fit=(s['ids'],s['parts'],[s['a'],s['b']],s['w']);pick=int(s['pick'])
    def apply(raw,p):
        q=prev(raw,p)
        for t in np.array_split(np.arange(len(q)),max(1,(len(q)+799)//800)):
            q[t,:,2]+=prediction(raw,t,fit)[0][:,:,pick]
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as ancestor
    previous,update=ancestor.load(),layer()
    return lambda raw,p:update(raw,previous(raw,p))


if __name__=='__main__':
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True)
    if run():incumbent.base.chain.replay.verify(OUT,layer,incumbent.OUT)
