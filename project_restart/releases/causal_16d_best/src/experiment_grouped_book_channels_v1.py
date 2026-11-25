"""Extend causal normalization to groups of previously untransported raw book channels."""
import json
import numpy as np
from scipy.cluster.hierarchy import linkage,to_tree
from scipy.spatial.distance import squareform

import experiment_prefix_book_transport_v1 as base
import experiment_transport_near_payoff_v1 as incumbent
import replay_periodic_parent_v1 as replay

clock=base.clock
OUT=clock.ART/'grouped_book_channels_v1'
FIELDS=np.setdiff1d(np.arange(83),[0,11,14,21,26,38,41,42,50,51,54,55,77,78])


def hierarchy():
    cov=[]
    for day in range(8):
        raw=clock.book.rawday(day)
        x=np.arcsinh(raw[day%13::13][:,:,FIELDS].astype('float32'))
        a=clock.center(x).reshape(-1,len(FIELDS)).astype(float)
        cov.append(a.T@a/len(a))
    g=np.mean(cov,axis=0)
    std=np.sqrt(g.diagonal()).clip(.01)
    distance=(1-abs(g/(std[:,None]*std))).clip(0,1)
    np.fill_diagonal(distance,0)
    tree=to_tree(linkage(squareform(distance,checks=False),method='average'))
    def split(branch):
        if branch.count<=10:return [FIELDS[branch.pre_order()]]
        return split(branch.left)+split(branch.right)
    groups=split(tree)
    # Small disconnected leaves share a group, limiting tiny single-feature readouts.
    small=[v for v in groups if len(v)<4];groups=[v for v in groups if len(v)>=4]
    if small:
        rest=np.concatenate(small)
        groups.extend(np.array_split(rest,max(1,(len(rest)+9)//10)))
    assert np.array_equal(np.sort(np.concatenate(groups)),FIELDS)
    return groups


def obs(raw,fields):
    a=clock.center(np.arcsinh(raw[:,:,fields].astype('float32')))
    total=np.concatenate((np.zeros_like(a[:1],dtype=float),a.cumsum(0,dtype=float)))
    t=np.arange(1,len(raw)+1);start=(t-24).clip(0)
    avg=((total[t]-total[start])/(t-start)[:,None,None]).astype('float32')
    return np.concatenate((a-avg,avg),axis=2)


def historical(groups):
    shape=[]
    for ids in groups:
        blocks=[]
        for day in range(8):
            y=obs(clock.book.rawday(day),ids)[day%11::11]
            a=y.reshape(-1,y.shape[2]).astype(float)
            blocks.append(a.T@a/len(a))
        g=np.mean(blocks,0);sd=np.sqrt(g.diagonal()).clip(.025)
        shape.append((ids,sd,g/(sd[:,None]*sd)))
    return shape


def views(raw,t,geom):
    ids,sd,cov=geom
    x=obs(raw,ids)/sd;n=len(sd)
    ref=.8*cov+.2*np.eye(n)
    root=base.matrix_power(ref,.5)
    result=[x[t].astype(np.float32)]+[np.empty((len(t),50,n),np.float32) for _ in range(2)]
    for b in np.unique(t//80):
        start=b*80;history=x[max(0,start-720):start:3]
        ii=np.flatnonzero(t//80==b);sample=x[t[ii]]
        for side in (0,1):
            a=history[:,side::2].reshape(-1,n)
            g=(a.T@a+2400*cov)/(len(a)+2400)
            g=.8*g+.2*np.eye(n)
            matrix=root@base.matrix_power(root@g@root,-.5)@root
            v=sample[:,1-side::2]
            idx=np.ix_(ii,np.arange(1-side,50,2))
            result[1][idx]=v*np.sqrt(ref.diagonal()/g.diagonal()).clip(.6,1.7)
            result[2][idx]=v@matrix
    return result


def design(a):
    z=np.tanh(.75*a)
    return clock.center(np.concatenate((a,z,z*abs(z)),axis=2))


def record(d,geoms):
    raw=clock.book.rawday(d);t=np.arange(d%7,len(raw)-60,7)
    y=clock.labels(d).reshape(-1,50,3)[t].astype(float)
    y[:,:,2]-=clock.RATIO*y[:,:,1];y=clock.center(y).reshape(-1,3)
    out=[]
    for geo in geoms:
        stats=[]
        for v in views(raw,t,geo):
            a=design(v).reshape(len(y),-1).astype(float)
            stats.append((a.T@a/len(a),a.T@y/len(a)))
        out.append(stats)
    return out


def change(raw,t,geoms,weights):
    scale=np.zeros((len(t),50,3),np.float64);cross=np.zeros_like(scale)
    for geo,w in zip(geoms,weights):
        a,b,c=(design(val)@coef for val,coef in zip(views(raw,t,geo),w))
        scale+=b-a;cross+=c-b
    return [(.4/len(geoms))*clock.center(v) for v in (scale,cross)]


def screen():
    geo=historical(hierarchy());hist=[];folds=[];days=[];far=[]
    for end in (8,10,12,14):
        hist.extend(record(d,geo) for d in range(len(hist),end))
        w=[base.fit(h) for h in zip(*hist)]
        grouped=[[[],[]],[[],[]]];tails=[[],[]]
        for d in (end,end+1):
            raw=clock.book.rawday(d);t=np.arange(d%3,len(raw),3)
            y=clock.labels(d).reshape(-1,50,3)[t]
            z=change(raw,t,geo,w)
            if d==end:
                ok=t<287
                for a,b in zip(z,change(raw[:287],t[ok],geo,w)):
                    np.testing.assert_allclose(a[ok],b,atol=3e-7,rtol=0)
            for j,ref in enumerate((clock.incumbent.fi.peer.reference,clock.incumbent.parent.fixed.earlier.early.baseline)):
                p=ref(d).reshape(-1,50,3)[t]
                for k,diff in enumerate(z):
                    grouped[j][k].append((p,y,diff))
                    if not j:
                        a=clock.center(p[:,:,2]-clock.RATIO*p[:,:,1]);v=clock.center(y[:,:,2]-clock.RATIO*y[:,:,1])
                        tails[k].append((a.ravel(),(a+diff[:,:,2]).ravel(),v.ravel()))
            days.append([clock.deltas(g[-1:]) for g in grouped[0]])
        folds.append([[clock.deltas(g) for g in gs] for gs in grouped])
        late=[]
        for g in tails:
            a,b,y=(np.concatenate(v) for v in zip(*g))
            late.append(float(np.corrcoef(b,y)[0,1]-np.corrcoef(a,y)[0,1]))
        far.append(late)
        print('window',end+1,folds[-1][0],flush=True)
    keep=(np.asarray(folds).min((0,1))>0)&((np.array(days)>0).sum(0)>=5)
    keep[:,2]&=np.mean(far,0)>0
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,days=days,far=far,
                                      groups=[v[0].tolist() for v in geo],keep=keep.tolist()),indent=2))
    if keep.any():
        hist.extend(record(d,geo) for d in (14,15))
        params={'keep':keep,'count':len(geo)}
        for j,(g,h) in enumerate(zip(geo,zip(*hist))):
            for k,v in zip(('ids','sd','cov','w'),(*g,base.fit(h))):params[f'{k}_{j}']=v
        np.savez(OUT/'model.npz',**params)
    return keep.any()


def layer():
    parent=incumbent.layer()
    geo=[];weights=[]
    with np.load(OUT/'model.npz') as s:
        keep=s['keep']
        for j in range(int(s['count'])):
            geo.append([s[f'{k}_{j}'] for k in ('ids','sd','cov')]);weights.append(s[f'w_{j}'])
    def apply(raw,p):
        q=parent(raw,p)
        for t in np.array_split(np.arange(len(q)),min(4,len(q))):
            for val,mask in zip(change(raw,t,geo,weights),keep):q[t]+=val*mask
        return q
    return apply


def load():
    import experiment_periodic_near_carry_v1 as baseline
    upstream=baseline.load()
    adjust=layer()
    return lambda raw,p: adjust(raw,upstream(raw,p))


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    if screen():replay.verify(OUT,layer,incumbent.OUT)
