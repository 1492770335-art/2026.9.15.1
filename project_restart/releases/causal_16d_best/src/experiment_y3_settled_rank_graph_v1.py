"""Update conditional pair spreads from stock baskets whose sixty-tick outcomes have settled."""
import json
import sys

import numpy as np
import joint_field_amplitude_references as history
import experiment_compounded_return_audit_v1 as tape
import y3_rank_spread_geometry as graph
import experiment_joint_field_state_amplitudes_v1 as parent

source,base = parent.source,parent.base
OUT = parent.OUT.parent/'y3_settled_rank_graph_v1'
BLOCK=128
ARMS=('incumbent','gap alone','wrong identities','same basket shape','opposite basket shape',
      'across basket agreement','across basket above gap','far interval agreement','recent above expanding')


def rank_reliability(p,t,order):
    sorted_scores=np.take_along_axis(p[t,:,2],order,axis=1)
    gap=np.diff(sorted_scores,axis=1)
    one=np.ones((len(t),1),dtype=gap.dtype)
    adjacent=np.minimum(np.column_stack((one,gap)),np.column_stack((gap,one)))
    # A nearly tied pair may swap graph positions under roundoff. Fade incident observations
    # and recovered node scores to zero at ties, where the position carries no information.
    node=adjacent.clip(0,.001)/.001
    edge=np.minimum(node[:,graph.LEFT],node[:,graph.RIGHT])
    return node,edge


def cells(p,t,order):
    i,j=order[:,graph.LEFT],order[:,graph.RIGHT]
    rows=np.arange(len(t))[:,None]
    gap=p[t][rows,j]-p[t][rows,i]
    def levels(v,lower,upper):
        # Linear interpolation across each boundary keeps category mass continuous.
        level=((v-lower)/.024+.5).clip(0,1)+((v-upper)/.024+.5).clip(0,1)
        lo=level.astype(np.int32)
        weight=level-lo
        return ((lo,1-weight),((lo+1).clip(0,2),weight))
    size=levels(gap[:,:,2],.13,.38)
    early=levels(gap[:,:,0],-.09,.09)
    middle=levels(gap[:,:,1],-.09,.09)
    return [(9*(3*graph.LEVEL+a)+3*b+c,u*v*w) for a,u in size for b,v in early for c,w in middle]


def order_at(p,t):
    return np.argsort(p[t,:,2],axis=1,kind='stable')


def edge_records(raw,p,geo):
    k,coef,row=geo
    path=tape.compound(raw,k)
    missing=np.vstack((np.zeros((1,50),int),(~tape.present(raw,row)).cumsum(0)))
    records=[]
    for end in range(BLOCK,len(raw)+1,BLOCK):
        t=np.arange(end-BLOCK,end-60,7)
        ticks=np.arange(len(t))[:,None]
        order=order_at(p,t)
        membership=cells(p,t,order)
        i,j=order[:,graph.LEFT],order[:,graph.RIGHT]
        _,reliability=rank_reliability(p,t,order)
        ok=missing[t+61]==missing[t]
        near=coef[1,0]+coef[1,1]*tape.span(path,t,20,k)
        full=coef[2,0]+coef[2,1]*tape.span(path,t,60,k)
        pred=np.stack((p[t,:,2],p[t,:,2]-base.RATIO*p[t,:,1]),2)
        actual=np.stack((full,full-base.RATIO*near),2)
        # State groups have distinct market exposures. Fit the scale on resolved relative spreads.
        pa=pred[ticks,j]-pred[ticks,i]
        ya=actual[ticks,j]-actual[ticks,i]
        valid=ok[ticks,i]&ok[ticks,j]
        scale=((pa*ya*valid[:,:,None]).sum((0,1))/(pa**2*valid[:,:,None]).sum((0,1)).clip(.01)).clip(.45,2.2)
        err=(ya/scale-pa).clip(-2.8,2.8)
        wrong=((actual[ticks,(j+11)%50]-actual[ticks,(i+11)%50])/scale-pa).clip(-2.8,2.8)
        ok_wrong=ok[ticks,(i+11)%50]&ok[ticks,(j+11)%50]&valid
        # Fit only edges lying entirely in one of two disjoint stock baskets.
        groups=[]
        for half in (0,1):
            side=(i%2==half)&(j%2==half)
            targets=[]
            for z,mask in ((err,valid),(wrong,ok_wrong)):
                use=side&mask
                vals=np.zeros((162,3))
                for ix,frac in membership:
                    weight=reliability[use]*frac[use]
                    bins=ix[use]
                    vals[:,0]+=np.bincount(bins,weights=weight,minlength=162)
                    for jh,zh in enumerate(z.transpose(2,0,1)):
                        vals[:,jh+1]+=np.bincount(bins,weights=zh[use]*weight,minlength=162)
                targets.append(vals)
            groups.append(targets)
        records.append(groups)
    return np.asarray(records)


def tables(records):
    n=records[...,:1]
    s=records[...,1:]
    upper=records.reshape(len(records),2,2,18,9,3).sum(4)
    coarse=upper[...,1:]/(upper[...,:1]+26)
    prior=np.repeat(coarse,9,axis=3)
    # Treat each resolved block as one repetition regardless of its number of close stock pairs.
    dev=(s-n*prior)/(n+12)
    def shrink(v):
        m=v.mean(0)
        se=v.std(0)/np.sqrt(len(v))
        return m*abs(m)/(abs(m)+se+.018)*len(v)/(len(v)+4.)
    return shrink(prior),shrink(dev)


def signals(raw,p,geo):
    z=np.zeros((len(p),50,len(ARMS)-1),np.float32)
    for lo,hi in base.segments(len(p)):
        v=p[lo:hi]
        records=edge_records(raw[lo:hi],v,geo)
        for start in range(BLOCK,len(v),BLOCK):
            past=records[:start//BLOCK]
            prior,dev=tables(past)
            _,recent=tables(past[-7:])
            t=np.arange(start,min(start+BLOCK,len(v)))
            order=order_at(v,t)
            membership=cells(v,t,order)
            def interpolate(table):
                return sum(table[ix]*frac[:,:,None] for ix,frac in membership)
            node,reliability=rank_reliability(v,t,order)
            i,j=order[:,graph.LEFT],order[:,graph.RIGHT]
            def lookup(arr,opposite=False):
                side=(i%2+j%2)*.5
                if opposite:side=1-side
                return interpolate(arr[0])*(1-side[:,:,None])+interpolate(arr[1])*side[:,:,None]
            def mix(a):return .55*a[:,:,0]+.45*a[:,:,1]
            opposite=lookup(dev[:,0],True)
            g=mix(lookup(prior[:,0],True))
            c=mix(opposite)
            def agree(a,b):return np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0)
            joint=interpolate(agree(dev[0,0],dev[1,0]))
            edges=(g,mix(lookup(dev[:,1],True)),mix(lookup(dev[:,0])),c,mix(joint),c-g,
                   agree(opposite[:,:,0],opposite[:,:,1]),mix(lookup(recent[:,0],True))-c)
            ranked=.32*node[:,:,None]*np.stack([(e*reliability)@graph.PROJECT for e in edges],2)
            ranked-=ranked.mean(1,keepdims=True)
            np.put_along_axis(z[lo+start:lo+t[-1]+1],order[:,:,None],ranked,axis=1)
    return z


def train():
    OUT.mkdir(exist_ok=True)
    geo=source.history.trial.curve.settled.historical_geometry()
    weights=np.ones(len(ARMS)-1)
    folds,daily=[],[]
    for day in range(8,16):
        if day%4==0:group=[]
        raw,refs,y=source.read_day(day)
        pairs=[]
        for p in refs:
            z=signals(raw,p,geo)
            if day==8:
                for n in (127,129,513,2530):
                    np.testing.assert_allclose(z[:n],signals(raw[:n],p[:n],geo),atol=1e-6,rtol=0)
            if day==8:
                n=955
                pp=p[:n].copy()
                pp+=np.random.default_rng(513).uniform(-6e-7,6e-7,pp.shape)
                np.testing.assert_allclose(z[:n],signals(raw[:n],pp,geo),atol=1e-4,rtol=0)
            pairs.append(base.records(p,y,z))
        group.append(pairs)
        daily.append([base.score(m,weights)[0] for m in pairs])
        if day%4==3:folds.append([base.score(m,weights) for m in np.sum(group,0)])
    f,ds=np.array(folds),np.array(daily)
    passed=[i for i in range(4,len(ARMS)) if f[:,:,:,i].min()>0 and (ds[:,:,i]>0).sum(0).min()>=5
            and (f[:,:,0,i].mean(0)>f[:,:,0,1:4].mean(0).max(1)).all()]
    pick=max(passed,key=lambda i:f[:,:,0,i].min()) if passed else 0
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=pick,folds=f.tolist(),daily=ds.tolist()),indent=2))
    print('selected',ARMS[pick],f[:,0,0].round(7).tolist(),flush=True)
    if pick:np.savez(OUT/'model.npz',*geo,pick=pick-1)
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        geo,pick=(float(f['arr_0']),f['arr_1'],f['arr_2']),int(f['pick'])
    return lambda raw,p:signals(raw,p,geo)[:,:,pick]


def layer():
    before,update=parent.layer(),correction()
    def apply(raw,initial):
        p=before(raw,initial)
        p[:,:,2]+=update(raw,p)
        return p
    return apply


def replay_with():
    verifier,prev=parent.replay_with()
    update=correction()
    def read(d,raw):
        p,h,n=prev(d,raw)
        p[:,:,2]+=update(raw,p)
        h[:,:,2]+=update(raw[:n],h)
        return p,h,n
    return verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
