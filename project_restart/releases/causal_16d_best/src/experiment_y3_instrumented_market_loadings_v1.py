"""Price stock-specific market exposure with two disjoint donor baskets and separate forecastable shocks."""
import json
import sys
import numpy as np

import y3_settled_spread_references as history

parent=history.incumbent
base=parent.base
OUT=parent.OUT.parent/'y3_instrumented_market_loadings_v1'
GROUPS=np.random.default_rng(313617).permutation(50).reshape(5,10)
ARMS=('incumbent','wrong receivers','return-based exposure','forecast-based exposure','two-basket predictable exposure',
      'predictable exposure shrunk to realized exposure','agreement of exposure estimates','increment over realized',
      'common effect stable across dates','surprise relative to trailing market')


def bands(p):
    return np.stack((p[:,:,2],p[:,:,2]-base.RATIO*p[:,:,1]),axis=2).astype(float)


def observations(p,y):
    a,b=bands(p),bands(y)
    slope=((a-a.mean((0,1)))*(b-b.mean((0,1)))).mean((0,1))/a.var((0,1))
    b/=slope.clip(.35,2.8)
    blocks=[]
    for k,ids in enumerate(GROUPS):
        donor1=GROUPS[[(k+1)%5,(k+3)%5]].ravel()
        donor2=GROUPS[[(k+2)%5,(k+4)%5]].ravel()
        q,r=[x[:,ids]-x[:,ids].mean(1,keepdims=True) for x in (a,b)]
        q-=q.mean(0)
        r-=r.mean(0)
        m,n=[np.stack((a[:,g].mean(1),b[:,g].mean(1)),2) for g in (donor1,donor2)]
        m-=m.mean(0)
        n-=n.mean(0)
        avg=(m+n)*.5
        # For each horizon the final dimension contains forecast, realized cross-products.
        val=[(r*avg[:,None,:,1]).mean(0),(q*avg[:,None,:,0]).mean(0),
             (r*avg[:,None,:,0]).mean(0),((r-q)*avg[:,None,:,0]).mean(0)]
        den=[(m[:,:,1]*n[:,:,1]).mean(0),(m[:,:,0]*n[:,:,0]).mean(0),
             (.5*(m[:,:,1]*n[:,:,0]+m[:,:,0]*n[:,:,1])).mean(0),
             (avg[:,:,0]**2).mean(0)]
        blocks.append((np.stack(val),np.stack(den)))
    return tuple(np.stack(v) for v in zip(*blocks))


def fit(rows):
    # Shapes are dates, recipient baskets, statistics, stocks (numerators only), horizons.
    numer,denom=map(np.stack,zip(*rows))
    def estimate(d):
        n,dn=numer[d].mean(0),denom[d].mean(0)
        def div(num,den,floor):
            return num/(den.clip(floor)[:,None,:]+floor)
        truth=div(n[:,0],dn[:,0],.025)
        priced=div(n[:,2],dn[:,2],.012)
        predicted=div(n[:,1],dn[:,1],.009)
        ordinary=div(n[:,3],dn[:,3],.009)
        # Convert the observable market forecast to the realized return units of each horizon.
        scale=(dn[:,2]/(dn[:,1].clip(.004)+.001)).clip(.35,2.5)
        realized=truth*scale[:,None,:]-predicted
        predictable=priced*scale[:,None,:]-predicted
        return np.stack((realized,ordinary,predictable),axis=0)
    coeff=estimate(slice(None))
    daily=np.stack([estimate([i]) for i in range(len(rows))])
    scale=daily.std(0)/np.sqrt(len(rows))+.012
    strength=abs(coeff)/(abs(coeff)+1.5*scale)
    realized,ordinary,priced=coeff*strength
    hybrid=(coeff[2]*strength[2]+realized*(1-strength[2]))*.75
    def agree(a,b):return np.where(a*b>0,np.sign(a)*np.minimum(abs(a),abs(b)),0)
    stable=agree(estimate(slice(0,None,2))[2],estimate(slice(1,None,2))[2])*strength[2]
    bank=np.stack((np.roll(priced,3,axis=1),realized,ordinary,priced,hybrid,agree(realized,priced),
                   priced-realized,stable,priced))
    return bank.clip(-1.2,1.2)


def signals(p,w):
    a=bands(p)
    out=np.empty((*p.shape[:2],len(w)),np.float32)
    for k,ids in enumerate(GROUPS):
        donor=GROUPS[np.arange(5)!=k].ravel()
        market=a[:,donor].mean(1)
        # A second view prices only the change in the observable common forecast.
        lag=np.empty_like(market)
        for lo,hi in base.segments(len(p)):
            u=market[lo:hi]
            cs=np.vstack((np.zeros((1,2)),u.cumsum(0)))
            end=np.arange(1,len(u)+1)
            left=(end-80).clip(0)
            lag[lo:hi]=(cs[end]-cs[left])/(end-left)[:,None]
        for i,v in enumerate(w[:,k]):
            m=market-lag if i==len(w)-1 else market
            out[:,ids,i]=.32*((m[:,None]*v)*np.array([.58,.42])).sum(2)
    return out-out.mean(1,keepdims=True)


def cosine(a,b):
    return float((a*b).sum()/np.sqrt((a*a).sum()*(b*b).sum()).clip(1e-8))


def train():
    OUT.mkdir(exist_ok=True)
    rows,folds,daily,audit=[],[],[],[]
    weights=np.ones(len(ARMS)-1)
    for d in range(4,16):
        if d in (8,12):
            bank=fit(rows)
            group=[]
            later=[]
        _,refs,y=history.training_day(d)
        stats=[observations(p,y) for p in refs]
        summary=tuple(np.mean(v,0) for v in zip(*stats))
        if d>=8:
            pairs=[]
            for p in refs:
                z=signals(p,bank)
                if d==8:
                    np.testing.assert_allclose(z[:2427],signals(p[:2427],bank),atol=1e-6,rtol=0)
                pairs.append(base.records(p,y,z))
            group.append(pairs)
            later.append(summary)
            daily.append([base.score(m,weights)[0] for m in pairs])
            if d in (11,15):
                folds.append([base.score(m,weights) for m in np.sum(group,0)])
                future=fit(later)
                audit.append({ARMS[i+1]:cosine(bank[i],future[i]) for i in (1,2,3,4)})
        rows.append(summary)
    f,ds=np.array(folds),np.array(daily)
    eligible=[i for i in range(4,len(ARMS)) if f[:,:,:,i].min()>0 and (ds[:,:,i]>0).sum(0).min()>=5
              and (f[:,:,0,i].mean(0)>f[:,:,0,1:4].mean(0).max(1)).all()]
    pick=max(eligible,key=lambda i:f[:,:,0,i].min()) if eligible else 0
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=pick,folds=f.tolist(),
                                                   daily=ds.tolist(),loading_cosines=audit),indent=2))
    print('selected',ARMS[pick],f[:,0,0].round(6).tolist(),audit,flush=True)
    if pick:np.savez(OUT/'model.npz',w=fit(rows),pick=pick-1)
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        w,pick=f['w'],int(f['pick'])
    return lambda p:signals(p,w)[:,:,pick]


def layer():
    previous,update=parent.layer(),correction()
    def apply(raw,initial):
        p=previous(raw,initial)
        p[:,:,2]+=update(p)
        return p
    return apply


def replay_with():
    verifier,previous=parent.replay_with()
    update=correction()
    def read(d,raw):
        p,h,n=previous(d,raw)
        p[:,:,2]+=update(p)
        h[:,:,2]+=update(h)
        return p,h,n
    return verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
