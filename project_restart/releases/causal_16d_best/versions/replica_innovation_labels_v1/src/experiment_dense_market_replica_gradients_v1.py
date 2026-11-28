"""Read market information from derivatives along two zero-net-investment portfolios that track the opposite basket."""
from functools import lru_cache
import json
import sys
import numpy as np

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip
import experiment_dense_market_weight_derivative_v1 as basis

ref,REFERENCE,read_p,TARGETS=basis.ref,basis.REFERENCE,basis.read_p,basis.TARGETS
OUT=REFERENCE.parent/'dense_market_replica_gradients_v1'
ARMS=('opposite_basket_cash','replica_and_market_agree','panels_agree')
SIDE=np.random.default_rng(94217).permutation(50)%2
N=6+basis.F


@lru_cache(maxsize=16)
def price_cov(day):
    raw,grid=[onset.settled.load(day,k) for k in ('x','time')]
    rows=[[],[]]
    for a,b,m,ok in event.sessions(raw,grid):
        t=np.arange(day%7,b-a-60,7)
        path=m.cumsum(0);miss=np.vstack((np.zeros((1,50),int),(~ok).cumsum(0)))
        for j,(s,h) in enumerate(((5,20),(20,60))):
            good=(miss[t+h+1]==miss[t]).all(1)
            rows[j].append((path[t+h]-path[t+s])[good])
    return np.stack([np.cov(np.concatenate(r),rowvar=False,bias=True) for r in rows])


@lru_cache(maxsize=1)
def replicas():
    daily=np.stack([price_cov(d) for d in range(4)])
    weights=np.zeros((2,2,50))
    for j in (0,1):
        for s in (0,1):
            i=np.flatnonzero(SIDE==s);o=np.flatnonzero(SIDE!=s)
            center=np.eye(len(i))-1/len(i)
            co=[]
            for g in daily[:,j]:
                inside=g[np.ix_(i,i)]
                cov=center@inside@center
                co.append(center@np.linalg.solve(cov+.45*np.diag(inside.diagonal()),center@g[np.ix_(i,o)].mean(1)))
            co=np.stack(co);mean=co.mean(0)
            w=center@(mean**3/(mean**2+co.var(0)/len(co)+1e-6))
            weights[j,s,i]=w
    return weights


def observations(x,p,live,norm,weights,with_targets=None):
    _,dx=basis.basis(x,live,norm)
    res=[]
    for j in range(2):
        panels=[]
        for k in (0,1):
            w=weights[j,k]*live
            # Recenter only the active names in the issuing panel to preserve zero cost.
            inside=live&(SIDE==k)
            w-=inside*(w.sum(1)/inside.sum(1).clip(1))[:,None]
            other=live&(SIDE!=k)
            wo=other/other.sum(1).clip(1)[:,None]
            hedge=np.einsum('ts,tsf->tf',w,dx)
            sp=w[:,:,None]*p
            pp=sp.sum(1)
            pm=(wo[:,:,None]*p).sum(1)
            design=np.column_stack((pm,pp,hedge))
            if with_targets is not None:
                label=with_targets[:,:,j]
                targets=np.stack(((wo*label).sum(1),(w*label).sum(1)),axis=-1)
                design=np.column_stack((design,targets))
            panels.append(design)
        res.append(panels)
    return res


@lru_cache(maxsize=12)
def record(day):
    raw,grid,y=[onset.settled.load(day,k) for k in ('x','time','y')]
    p=read_p(day)
    g=np.zeros((2,2,N+2,N+2));count=np.zeros(2)
    for a,b,_,ok in event.sessions(raw,grid):
        x=basis.stock_state(raw[a:b],ok)
        t=np.arange(30+day%7,b-a,7)
        for block in np.array_split(t,max(1,len(t)//48)):
            parts=observations(x[block],p[a+block],ok[block],basis.normalization(),replicas(),y[a+block,:,1:])
            for j,panels in enumerate(parts):
                for k,u in enumerate(panels):
                    u-=u.mean(0)
                    g[j,k]+=u.T@u
                count[j]+=len(block)
    return g/count[:,None,None,None]


def train(end):
    ds=np.stack([record(d) for d in range(4,end)]).mean(2)
    weights=[]
    for j in (0,1):
        rows=ds[:,j];g=rows.mean(0)
        proj=np.eye(N)[:,6:]
        ctrl=g[:6,:6]
        proj[:6]=-np.linalg.solve(ctrl+.075*np.diag(ctrl.diagonal().clip(1e-7)),g[:6,6:N])
        sub=proj.T@g[:N,:N]@proj
        sd=sub.diagonal().clip(1e-8)**.5
        ev,u=np.linalg.eigh(sub/np.outer(sd,sd))
        ax=proj@(u/(sd[:,None]*np.sqrt(ev.clip(0)+.85)))
        slopes=[]
        for k,pi in enumerate((1+j,4+j)):
            beta=(rows[:,pi,N+k]/rows[:,pi,pi].clip(1e-9)).clip(.2,4)
            target=rows[:,:N,N+k]/beta[:,None]-rows[:,:N,pi]
            v=target@ax;mean=v.mean(0)
            slopes.append(mean**3/(mean**2+2*v.var(0)/max(len(v)-1,1)+1e-9))
        direct,replicated=slopes
        match=direct*replicated>0
        weights.append(np.stack((ax@direct,ax@(direct*match)),axis=1))
    return np.r_[basis.normalization().ravel(),replicas().ravel(),np.stack(weights).ravel()]


def signals(raw,grid,p,bank):
    offset=2*basis.C
    norm=bank[:offset].reshape(2,basis.C)
    w=bank[offset:offset+200].reshape(2,2,50)
    co=bank[offset+200:].reshape(2,N,2)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,_,ok in event.sessions(raw,grid):
        x=basis.stock_state(raw[a:b],ok)
        for lo in range(0,b-a,64):
            t=np.arange(lo,min(b-a,lo+64))
            parts=observations(x[t],p[a+t],ok[t],norm,w)
            for j,pair in enumerate(parts):
                left,right=[q@co[j] for q in pair]
                consensus=np.where(left[:,1]*right[:,1]>0,np.sign(left[:,1])*np.minimum(abs(left[:,1]),abs(right[:,1])),0.)
                out[a+t,:,:,j]=np.column_stack(((left+right)*.5,consensus))[:,None]
    return out


def audit():
    w=replicas();out={}
    for end in (8,12):
        g=np.mean([price_cov(d) for d in range(end,end+4)],axis=0)
        result=[]
        for j in (0,1):
            vals=[]
            for k in (0,1):
                other=(SIDE!=k)/25
                v=w[j,k]
                vals.append(float((v@g[j]@other)/np.sqrt((v@g[j]@v)*(other@g[j]@other))))
            result.append(vals)
        out[end+1]=result
    (OUT/'disjoint_market_tracking.json').write_text(json.dumps(out,indent=2))
    print(out,flush=True)


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    strip.evaluation.rolling(sys.modules[__name__])
    audit()
