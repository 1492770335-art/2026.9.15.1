"""Track whether successive revisions for a single terminal price continue or reverse."""
import json
import numpy as np
import experiment_y3_settled_factor_confidence_v1 as incumbent
import experiment_dominant_far_block_v1 as metric
import replay_periodic_parent_v1 as replay

clock=incumbent.clock
OUT=clock.ART/'y3_terminal_revision_chain_v1'
UNIT=np.array([.558942,.272164,.160943])


def lag(x,h):
    return x[(np.arange(len(x))-h).clip(0)]


def trailing(x,n):
    c=np.concatenate((np.zeros_like(x[:1],dtype=float),x.cumsum(0,dtype=float)))
    t=np.arange(1,len(c));a=(t-n).clip(0)
    return (c[t]-c[a])/(t-a).reshape((-1,)+(1,)*(x.ndim-1))


def streams(raw,p):
    price=(raw[:,:,0].astype(float)-.00136).cumsum(0)*UNIT[2]
    prefix=p*(UNIT[2]/UNIT)
    states=[]
    for long,short,delay in ((2,1,40),(1,0,15),(0,-1,5)):
        end=prefix[:,:,short] if short>=0 else 0
        change=price-lag(price,delay)+end-lag(prefix[:,:,long],delay)
        change[:delay]=0.
        states.append(clock.center(change))
    return states,clock.center(prefix)


def design(raw,p,aligned):
    states,cur=streams(raw,p)
    a,b,c=states
    cols=[cur, np.tanh(3*cur)]
    for v in states:cols.append(np.tanh(v)[:,:,None])
    context=[]
    for first,second,h in ((a,b,15),(b,c,5),(a,c,20)):
        # A,B,C end at remaining maturities 20,5,0. Only the exact separation joins the same endpoint.
        older=lag(first,h if aligned else 2*h)
        # Remove slowly varying directional means before estimating persistence.
        older=older-trailing(older,160)
        later=second-trailing(second,160)
        for width in (96,384):
            covariance=trailing(older*later,width)
            scale=np.sqrt((.015+trailing(older**2,width))*(.015+trailing(later**2,width)))
            rho=(covariance/scale).clip(-1,1)
            centered=rho-rho.mean(1,keepdims=True)
            context.extend((centered*np.tanh(first),centered*np.tanh(second),
                            centered*np.tanh(4*cur[:,:,2]),rho*np.tanh(3*(cur[:,:,2]-cur[:,:,1]))))
    return clock.center(np.concatenate((*cols,np.stack(context,2)),2)).astype('float32')


def stats(x,p,y):
    targets=[]
    for a,b in ((p[:,:,2],y[:,:,2]),(p[:,:,2]-clock.RATIO*p[:,:,1],y[:,:,2]-clock.RATIO*y[:,:,1])):
        a,b=(clock.center(v.astype(float)) for v in (a,b))
        slope=((a*b).mean()/(a*a).mean().clip(1e-8)).clip(.4,2.)
        targets.append((b/slope-a).clip(-5,5))
    flat=x.reshape(-1,x.shape[2]).astype(float)
    return flat.T@flat/len(flat),flat.T@np.stack(targets,2).reshape(-1,2)/len(flat)


def fit(hist):
    result=[]
    for arm in zip(*hist):
        g,b=(np.stack(s) for s in zip(*arm));mean=g.mean(0)
        penalty=.9*np.diag(mean.diagonal().clip(.00004))+.00003*np.eye(len(mean))
        pooled=np.linalg.solve(mean+penalty,b.mean(0))
        daily=np.linalg.solve(g+penalty,b)
        result.append(.7*pooled+.3*np.median(daily,0))
    return np.stack(result)


def change(views,w):
    return .24*clock.center(views[1]@w[1]-views[0]@w[0])


def run():
    history=[];folds=[];days=[]
    for end in (8,10,12,14):
        weights=fit(history) if history else None
        groups=[[[],[]],[[],[]]]
        for d in (end,end+1):
            raw=clock.book.rawday(d)
            y=clock.labels(d).reshape(-1,50,3)
            for j,ref in enumerate(incumbent.REFS):
                p=ref(d).reshape(y.shape)
                views=[design(raw,p,flag) for flag in (False,True)]
                if j==0:
                    t=np.arange(d%5,len(raw)-60,5)
                    history.append([stats(x[t],p[t],y[t]) for x in views])
                if weights is not None:
                    z=change(views,weights)
                    if not j and d==end:
                        n=733
                        short=[design(raw[:n],p[:n],flag) for flag in (False,True)]
                        np.testing.assert_allclose(change(short,weights),z[:n],atol=1e-8,rtol=0)
                    for k in (0,1):groups[j][k].append((p,y,z[:,:,k]))
            if weights is not None:days.append([metric.gain(g[-1:]) for g in groups[0]])
        if weights is not None:
            folds.append([[metric.gain(g) for g in side] for side in groups])
            print(end+1,folds[-1][0],flush=True)
    eligible=[k for k in range(2) if all(sum(r[j][k]['y3']>0 for r in folds)>=2 and folds[-1][j][k]['y3']>0 and
              np.mean([r[j][k]['y3'] for r in folds])>0 and np.mean([r[j][k]['relative_far'] for r in folds])>0
              for j in (0,1)) and sum(d[k]['y3']>0 for d in days)>=4]
    chosen=max(eligible,key=lambda k:np.mean([r[0][k]['y3'] for r in folds])) if eligible else None
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=days,chosen=chosen,
        targets=['full Y3 residual','far residual'],windows=[[11,12],[13,14],[15,16]]),indent=2))
    if chosen is not None:np.save(OUT/'weights.npy',fit(history)[:,:,chosen:chosen+1])
    return chosen is not None


def layer():
    parent=incumbent.layer();w=np.load(OUT/'weights.npy')
    def predict(raw,p):
        q=parent(raw,p)
        z=change([design(raw,q,v) for v in (False,True)],w)
        q[:,:,2]+=z[:,:,0]
        return q
    return predict


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if run():replay.verify(OUT,layer,incumbent.OUT)
