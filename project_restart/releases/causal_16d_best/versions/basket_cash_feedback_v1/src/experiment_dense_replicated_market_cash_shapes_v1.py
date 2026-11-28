"""Estimate horizon shapes using cross-products of independent daily market-return regressions."""
import sys
import numpy as np
import experiment_dense_market_forward_cash_v1 as cash

strip,rel,onset,REFERENCE,read_p=cash.strip,cash.rel,cash.onset,cash.REFERENCE,cash.read_p
OUT=REFERENCE.parent/'dense_replicated_market_cash_shapes_v1'
ARMS=('stable_horizon_shape','noise_adjusted_shape','replicated_distal_shape')
N=12
C=7
W=C+4*cash.study.W


def past_returns(raw,grid):
    k,coef,bad=strip.physical()
    out=np.zeros((len(raw),4))
    for a,b in rel.fields.parent.sessions(grid):
        x=raw[a:b]
        ok=~np.all(x==bad,axis=-1)
        r=np.log1p(k*(x[...,0].astype(float)-.00136))*coef[2]/k
        price=((r*ok).sum(1)/ok.sum(1).clip(1)).cumsum()
        t=np.arange(b-a)
        out[a:b]=np.column_stack([price-price[(t-h).clip(0)] for h in (5,20,60,240)])
    return out


def fields(x,p,ret):
    return np.column_stack((p.mean(1,dtype=float),ret,x.reshape(len(x),-1)))


def collect():
    OUT.mkdir(exist_ok=True)
    _,coef,_=strip.physical()
    for d in range(4,16):
        x,legs,_,t=cash.records(d)
        raw,grid,y=[onset.settled.load(d,k) for k in ('x','time','y')]
        p=read_p(d)
        u=p.astype(float)-p.mean((0,1),dtype=float)
        v=y.astype(float)-y.mean((0,1),dtype=float)
        beta=(u*v).mean((0,1))/(u*u).mean((0,1))
        curve=(p[t].mean(1,dtype=float)*(beta*coef[2]/coef))@strip.INTERP
        error=legs-np.diff(curve,axis=-1,prepend=np.zeros((len(t),1)))
        f=fields(x,p[t],past_returns(raw,grid)[t])
        row=np.column_stack((f,error))
        mean=row.mean(0)
        row-=mean
        np.savez(OUT/f'day_{d:02d}.npz',gram=row.T@row/len(row),mean=mean[:W],beta=beta)


def shrink_horizons(g):
    n=len(g)
    even,odd=g[::2].mean(0),g[1::2].mean(0)
    # Each side estimates cash loadings independently. Noise within a date may couple many leads.
    signal=(even.T@odd+odd.T@even)/2
    scatter=g-g.mean(0)
    noise=np.einsum('dri,drj->ij',scatter,scatter)/(n*(n-1))
    floor=max(np.trace(noise)/N*.15,1e-9)
    noise+=floor*np.eye(N)
    val,u=np.linalg.eigh(noise)
    root=(u*np.sqrt(val))@u.T
    inv=(u/np.sqrt(val))@u.T
    lam,v=np.linalg.eigh(inv@signal@inv)
    w=lam.clip(0)/(1+lam.clip(0))
    return inv@(v*w)@v.T@root


def train(end):
    rows=[];means=[];betas=[]
    for d in range(4,end):
        with np.load(OUT/f'day_{d:02d}.npz') as f:
            rows.append(f['gram']);means.append(f['mean']);betas.append(f['beta'])
    rows=np.stack(rows)
    g=rows.mean(0)
    proj=np.eye(W)[:,C:]
    proj[:C]=-np.linalg.solve(g[:C,:C]+.015*np.diag(g.diagonal()[:C].clip(1e-6)),g[:C,C:W])
    cov=proj.T@g[:W,:W]@proj
    sigma=np.sqrt(cov.diagonal().clip(1e-6))
    eig,U=np.linalg.eigh(cov/np.outer(sigma,sigma))
    basis=proj@(U[:,-48:]/(sigma[:,None]*np.sqrt(eig[-48:].clip(0)+1.5)))
    cross=basis.T@rows[:,:W,W:]
    # Learn the repeatable return shape on one half of feature coordinates and apply it to the other.
    model=np.zeros((48,3,N))
    for parity in (0,1):
        learn=np.arange(1-parity,48,2)
        apply=np.arange(parity,48,2)
        gain=shrink_horizons(cross[:,learn])
        data=cross[:,apply]
        avg=data.mean(0)
        stable=avg**3/(avg**2+data.var(0)/len(rows)+1e-9)
        model[apply,0]=stable@gain
        model[apply,1]=avg@gain
        aa,bb=[z.mean(0)@gain for z in (data[::2],data[1::2])]
        model[apply,2]=np.where(aa*bb>0,np.sign(aa)*np.minimum(abs(aa),abs(bb)),0.)
    _,coef,_=strip.physical()
    units=np.mean(betas,0)*coef[2]/coef
    weights=np.empty((W,3,2))
    for h,(l,r) in enumerate(((1,4),(4,12))):
        for j in range(3):
            weights[:,j,h]=basis@model[:,j,l if j==2 else 0:r].sum(-1)/units[h+1]
    return np.r_[np.mean(means,axis=0),weights.ravel()]


def signals(raw,grid,p,bank):
    mean=bank[:W]
    w=bank[W:].reshape(W,6)
    past=past_returns(raw,grid)
    out=np.empty((len(p),6))
    for a,b in rel.fields.parent.sessions(grid):
        market=cash.study.clock.summary(raw[a:b])
        for i in range(0,len(market),64):
            t=np.arange(i,min(i+64,len(market)))
            x=cash.study.design(market[t].astype(np.float16),cash.study.clock.histories(market,t))
            out[a+t]=(fields(x,p[a+t],past[a+t])-mean)@w
    return np.broadcast_to(out.reshape(len(p),1,3,2),(*p.shape[:2],3,2))


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    collect()
    strip.evaluation.rolling(sys.modules[__name__])
