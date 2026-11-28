"""Fit a basket return function and supervise its derivatives under infinitesimal changes to stock weights."""
from functools import lru_cache
import json
import sys
import numpy as np

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip
import experiment_dense_replicable_maturity_modes_v1 as prior
import experiment_dense_paired_pressure_clock_v1 as book

ref,REFERENCE,read_p=prior.ref,prior.REFERENCE,prior.read_p
OUT=REFERENCE.parent/'dense_market_weight_derivative_v1'
ARMS=('market_values','basket_derivatives','combined_value_and_derivative')
TARGETS=(1,2)
SIGNED=(0,11,26,34,36,38,64)
K=12
C=4*K
H=36
F=C+2*H
rng=np.random.default_rng(514271)
W=rng.normal(size=(C,H))/np.sqrt(C)
BIAS=rng.uniform(-.9,.9,size=H)


def stock_state(raw,ok):
    cols=np.tanh(.65*np.arcsinh(raw[...,list(SIGNED)].astype(float)))
    paired,_=book.pressure(raw,ok)
    v=np.concatenate((cols,paired),axis=2)
    pre=np.concatenate((np.zeros_like(v[:1]),(v*ok[...,None]).cumsum(0)))
    num=np.vstack((np.zeros(50),ok.cumsum(0)))
    t=np.arange(len(v))+1
    def average(lo,hi):
        a=(t-hi).clip(0);b=(t-lo).clip(0)
        return (pre[b]-pre[a])/(num[b]-num[a]).clip(1)[...,None]
    return np.concatenate((average(0,4),average(4,24),average(24,96)),axis=2)


def coordinates(x,live):
    w=live/live.sum(1).clip(1)[:,None]
    mean=np.einsum('ts,tsf->tf',w,x)
    delta=x-mean[:,None]
    squared=delta[:,:,:K]**2
    var=np.einsum('ts,tsf->tf',w,squared)
    state=np.concatenate((mean,var),axis=1)
    # Derivative of the variance under a shift from the uniform basket to one stock.
    tangent=np.concatenate((delta,squared-var[:,None]),axis=2)
    return state,tangent


@lru_cache(maxsize=1)
def normalization():
    states=[];local=[]
    for d in range(4):
        raw,grid=[onset.settled.load(d,k) for k in ('x','time')]
        for a,b,_,ok in event.sessions(raw,grid):
            t=np.arange(96+d,b-a,29)
            x=stock_state(raw[a:b],ok)
            v,dv=coordinates(x[t],ok[t])
            states.append(v);local.append((dv**2).mean((0,1)))
    v=np.concatenate(states)
    return np.stack((v.mean(0),(v.var(0)+.04*np.mean(local,axis=0)+.0001)**.5))


def basis(x,live,norm,with_derivative=True):
    v,dv=coordinates(x,live)
    u=(v-norm[0])/norm[1]
    z=np.tanh(u@W+BIAS)
    features=np.concatenate((u,z,z*z),axis=1)
    if not with_derivative:
        return features,None
    du=dv/norm[1]
    dz=(du@W)*(1-z[:,None]**2)
    return features,np.concatenate((du,dz,2*z[:,None]*dz),axis=2)


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y=[onset.settled.load(d,k) for k in ('x','time','y')]
    p=read_p(d)
    shape=F+6
    gs=np.zeros((2,shape,shape))
    counts=np.zeros(2)
    for a,b,m,ok in event.sessions(raw,grid):
        x=stock_state(raw[a:b],ok)
        ticks=np.arange(25+d%7,b-a,7)
        for block in np.array_split(ticks,max(1,len(ticks)//48)):
            live=ok[block]
            w=live/live.sum(1).clip(1)[:,None]
            v,dv=basis(x[block],live,normalization())
            pq=p[a+block].astype(float);labels=y[a+block,:,1:]
            pm=np.einsum('ts,tsj->tj',w,pq);ym=np.einsum('ts,tsj->tj',w,labels)
            market=np.column_stack((np.ones(len(block)),pm,v,ym))
            individual=np.concatenate((np.zeros((*live.shape,1)),pq-pm[:,None],dv,labels-ym[:,None]),axis=2)[live]
            for j,z in enumerate((market,individual)):
                gs[j]+=z.T@z;counts[j]+=len(z)
    return gs/counts[:,None,None]


def fit(ds,j):
    gm=ds[:,0].mean(0);gd=ds[:,1].mean(0)
    # Both equations have the same output units; balance their design energies.
    scale=np.trace(gm[4:-2,4:-2])/np.trace(gd[4:-2,4:-2]).clip(1e-9)
    out=[]
    for weight in (0.,1.,.5):
        g=(1-weight)*gm+weight*scale*gd
        # The derivative of an intercept is zero.
        ctrl=np.arange(4) if weight<1 else np.arange(1,4)
        proj=np.eye(F+4)[:,4:]
        cov=g[np.ix_(ctrl,ctrl)]
        proj[ctrl]=-np.linalg.solve(cov+.01*np.diag(cov.diagonal().clip(1e-8)),g[np.ix_(ctrl,np.arange(4,F+4))])
        sub=proj.T@g[:-2,:-2]@proj
        sd=sub.diagonal().clip(1e-7)**.5
        eig,vec=np.linalg.eigh(sub/np.outer(sd,sd))
        ax=proj@(vec/(sd[:,None]*np.sqrt(eig.clip(0)+1.5)))
        matrices=(1-weight)*ds[:,0]+weight*scale*ds[:,1]
        beta=(matrices[:,1+j+1,-2+j]/matrices[:,1+j+1,1+j+1].clip(1e-8)).clip(.2,4)
        payoff=matrices[:,:-2,-2+j]/beta[:,None]-matrices[:,:-2,2+j]
        cross=payoff@ax
        mu=cross.mean(0)
        co=ax@(mu**3/(mu**2+2*cross.var(0)/max(1,len(ds)-1)+1e-8))
        # Transfer a gradient estimate to the full basket with an intercept centered on training markets.
        co[0]-=gm[0,:-2]@co
        out.append(co)
    return np.stack(out,axis=1)


def train(end):
    ds=np.stack([record(d) for d in range(4,end)])
    return np.r_[normalization().ravel(),np.stack([fit(ds,j) for j in (0,1)],axis=-1).ravel()]


def signals(raw,grid,p,bank):
    norm=bank[:2*C].reshape(2,C)
    co=bank[2*C:].reshape(F+4,6)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,_,ok in event.sessions(raw,grid):
        x=stock_state(raw[a:b],ok)
        for lo in range(0,b-a,80):
            t=np.arange(lo,min(b-a,lo+80))
            v,_=basis(x[t],ok[t],norm,False)
            pm=(p[a+t]*ok[t,...,None]).sum(1)/ok[t].sum(1).clip(1)[:,None]
            z=np.column_stack((np.ones(len(t)),pm,v))@co
            out[a+t]=z.reshape(len(t),1,3,2)
    return out


def check_derivative():
    d=8
    raw,grid=[onset.settled.load(d,k) for k in ('x','time')]
    a,b,_,ok=next(iter(event.sessions(raw,grid)))
    ts=np.arange(120,640,20)
    x=stock_state(raw[a:b],ok)[ts]
    val,tangent=basis(x,ok[ts],normalization())
    wd=ok[ts]/ok[ts].sum(1,keepdims=True)
    eps=1e-6
    name=7
    mixed=(1-eps)*wd
    mixed[:,name]+=eps
    m=np.einsum('ts,tsf->tf',mixed,x)
    v=np.einsum('ts,tsf->tf',mixed,(x[:,:,:K]-m[:,None,:K])**2)
    u=(np.column_stack((m,v))-normalization()[0])/normalization()[1]
    z=np.tanh(u@W+BIAS)
    numerical=(np.column_stack((u,z,z*z))-val)/eps
    delta=float(abs(numerical-tangent[:,name]).max())
    assert delta<.001,delta
    report={'max_absolute_gradient_error':delta,'rows':len(ts),'features':F}
    (OUT/'derivative_check.json').write_text(json.dumps(report,indent=2))


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    strip.evaluation.rolling(sys.modules[__name__])
    check_derivative()
