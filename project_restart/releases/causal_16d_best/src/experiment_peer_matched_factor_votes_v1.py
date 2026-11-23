"""Test robust book-change ranks within stocks that have similar incumbent forecasts."""
import json
import numpy as np
from scipy.stats import rankdata
import experiment_dominant_far_block_v1 as base
import experiment_far_shock_followthrough_v1 as follow

peer=base.peer
OUT=peer.ART/'peer_matched_factor_votes_v1'
LAGS=(3,16,64)


def center(x):
    return x-x.mean(1,keepdims=True)


def field_changes(raw,t,field,lag):
    x=np.arcsinh(raw[t,:,field].astype('float32'))
    past=np.arcsinh(raw[(t-lag).clip(0),:,field].astype('float32'))
    return x-past


def scan(d):
    raw=base.paths.book.rawday(d)
    t=np.arange(64+d%9,len(raw)-60,9)
    y=peer.labels(d).reshape(-1,50,3)[t]
    far=center(y[:,:,2]-base.RATIO*y[:,:,1])
    stats=[]
    for lag in LAGS:
        current=np.arcsinh(raw[t].astype('float32'))
        old=np.arcsinh(raw[t-lag].astype('float32'))
        x=rankdata(current-old,axis=1).astype('float32')-25.5
        x=x.reshape(-1,100)
        stats.extend((x.T@far.ravel()/np.sqrt(np.maximum((x*x).sum(0),1.)*(far**2).sum())).tolist())
    return np.array(stats)


def select():
    scores=np.stack([scan(d) for d in range(8)])
    means=np.median(scores,axis=0)
    strength=abs(means)*abs(np.sign(scores).mean(0))**2
    chosen=[]
    for i in np.argsort(-strength):
        f,l=i%100,LAGS[i//100]
        if f not in [c[0] for c in chosen]:
            chosen.append((int(f),int(l),int(np.sign(means[i]))))
        if len(chosen)==12:break
    return chosen,scores


def votes(raw,p,t,chosen):
    q=p[t].astype(float)
    # Matching on short and long forecasts accounts for stock differences already priced by the model.
    rel=center(q)
    rel/=rel.std(1,keepdims=True).clip(.045)
    dist=np.sum((rel[:,:,None,:]-rel[:,None,:,:])**2,axis=3)
    idx=np.arange(50)
    dist[:,idx,idx]=np.inf
    near=np.argpartition(dist,9,axis=2)[:,:,:10]
    batch=np.arange(len(t))[:,None,None]
    x=np.stack([field_changes(raw,t,f,l)*s for f,l,s in chosen],axis=2)
    glob=(rankdata(x,axis=1).astype('float32')-25.5)/24.5
    cols=[]
    for low in range(0,len(t),100):
        sl=slice(low,low+100)
        peers=x[batch[sl],near[sl]]
        cols.append(np.sign(x[sl,:,None,:]-peers).mean(2))
    local=center(np.concatenate(cols))
    return glob,local


def stats(x,p,y):
    b=y[:,:,2].astype(float)
    a=p[:,:,2].astype(float)
    a-=a.mean();b-=b.mean()
    slope=np.clip((a*b).sum()/(a*a).sum(),.4,2.5)
    e=center(b/slope-a)
    g=x.reshape(-1,x.shape[-1]).astype(float)
    return g.T@g/len(g),g.T@e.ravel()/len(g)


def fit(history):
    grams,rhs=(np.stack(v) for v in zip(*history))
    g=grams.mean(0)
    reg=1.25*np.diag(g.diagonal().clip(.001))
    daily=np.linalg.solve(grams+reg,rhs[:,:,None])[:,:,0]
    w=.5*np.linalg.solve(g+reg,rhs.mean(0))+.5*np.median(daily,axis=0)
    return .65*w*abs(np.sign(daily).mean(0))


def change(raw,p,t,chosen,w):
    _,local=votes(raw,p,t,chosen)
    return center(local@w).astype('float32')


def screen():
    chosen,initial=select()
    histories=[[],[]];daily=[];folds=[];diagnostics=[]
    for start in (8,10,12,14):
        ws=[fit(hist) for hist in histories] if histories[0] else None
        groups=[[],[]]
        for d in (start,start+1):
            raw=base.paths.book.rawday(d)
            t=np.arange(d%4,len(raw),4)
            p=peer.reference(d);y=peer.labels(d).reshape(p.shape)[t]
            g,l=votes(raw,p,t,chosen)
            if d==start:
                st=t[t<191]
                for a,b in zip((g,l),votes(raw[:191],p[:191],st,chosen)):
                    np.testing.assert_allclose(a[:len(st)],b,rtol=0,atol=1e-7)
            far=center(y[:,:,2]-base.RATIO*y[:,:,1])
            def ic(x,target):
                x=x.reshape(-1,x.shape[-1])
                target=target.ravel()
                return (x.T@target/np.sqrt((x*x).sum(0)*(target*target).sum())).tolist()
            e=center(y[:,:,2]-p[t,:,2])
            diagnostics.append(dict(date=d+1,raw_far_ic=ic(g,far),matched_resid_ic=ic(l,e)))
            for i,x in enumerate((g,l)):
                histories[i].append(stats(x,p[t],y))
                if ws is not None:
                    z=center(x@ws[i])
                    groups[i].append((p[t],y,z))
            if ws is not None:daily.append([base.gain(k[-1:]) for k in groups])
        if ws is not None:
            folds.append([base.gain(k) for k in groups])
            print(start+1,folds[-1],flush=True)
    f=np.array([[s['y3'] for s in row] for row in folds])
    keep=f[:,1].min()>0 and (f[:,1]-f[:,0]).mean()>0 and sum(row[1]['y3']>0 for row in daily)>=4
    report=dict(chosen=chosen,first8_far_ic=[initial[:,LAGS.index(l)*100+f].tolist() for f,l,_ in chosen],
                held_days=diagnostics,folds=folds,daily=daily,retained=bool(keep))
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    if keep:np.savez(OUT/'model.npz',chosen=chosen,weight=fit(histories[1]))
    return keep


def load():
    parent=base.load()
    with np.load(OUT/'model.npz') as f:chosen,w=f['chosen'],f['weight']
    def predict(raw,p):
        q=parent(raw,p).copy()
        for start in range(0,len(raw),300):
            t=np.arange(start,min(start+300,len(raw)))
            q[t,:,2]+=change(raw,q,t,chosen,w)
        return q
    return predict


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():follow.replay_candidate(OUT,load)
