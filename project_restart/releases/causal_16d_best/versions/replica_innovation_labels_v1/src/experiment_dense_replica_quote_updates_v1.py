"""Infer market revisions from the quoted prices of two disjoint, market-tracking long-short portfolios."""
from functools import lru_cache
import json
import sys
import numpy as np

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip
import experiment_dense_market_replica_gradients_v1 as replica
import audit_dense_horizon_leg_noise_v1 as geometry

ref,REFERENCE,read_p,TARGETS=replica.ref,replica.REFERENCE,replica.read_p,replica.TARGETS
OUT=REFERENCE.parent/'dense_replica_quote_updates_v1'
ARMS=('static_replica_quotes','live_tracking_revision','independent_quote_agreement')
BLOCK=80
D=9


def hedge(g,side):
    i=np.flatnonzero(replica.SIDE==side);o=np.flatnonzero(replica.SIDE!=side)
    it=np.eye(len(i))-1/len(i)
    mat=g[np.ix_(i,i)]
    return it@np.linalg.solve(it@mat@it+.60*np.diag(mat.diagonal().clip(1e-6)),it@g[np.ix_(i,o)].mean(1))


@lru_cache(maxsize=1)
def priors():
    return np.mean([replica.price_cov(d) for d in range(4)],axis=0)


def session(m,ok,p,initial,code):
    q=geometry.physical(p,code)
    legs=np.diff(q,axis=2,prepend=np.zeros_like(q[:,:,:1]))
    path=m.cumsum(0);gap=np.vstack((np.zeros((1,50),int),(~ok).cumsum(0)))
    fixed=[[hedge(initial[j],k) for k in (0,1)] for j in (0,1)]
    running=initial*32
    data=np.zeros((len(m),2,2,D))
    for lo in range(0,len(m),BLOCK):
        hi=min(lo+BLOCK,len(m))
        for j,h in enumerate((15,40)):
            for k in (0,1):
                group=replica.SIDE==k
                start=fixed[j][k]
                live=hedge(running[j],k)
                msk=ok[lo:hi,group];w=msk/msk.sum(1,keepdims=True).clip(1)
                x=legs[lo:hi,group]
                avg=np.einsum('ts,tsf->tf',w,x)
                z=(x-avg[:,None])*msk[...,None]
                old=np.einsum('tsf,s->tf',z,start)
                change=np.einsum('tsf,s->tf',z,live-start)
                other=ok[lo:hi,~group]
                obs=(p[lo:hi,~group]*other[...,None]).sum(1)/other.sum(1).clip(1)[:,None]
                data[lo:hi,j,k]=np.column_stack((obs,old,change))
            stops=np.arange(max(h,lo),hi,max(5,h//2))
            issue=stops-h
            good=(gap[stops+1]==gap[issue]).all(1)
            dp=(path[stops]-path[issue])[good]
            if len(dp):
                running[j]=.94*running[j]+dp.T@dp
    return data


@lru_cache(maxsize=12)
def moment(day):
    raw,grid,y=[onset.settled.load(day,k) for k in ('x','time','y')]
    p=read_p(day);g=np.zeros((2,D+3,D+3));n=0.
    for a,b,m,ok in event.sessions(raw,grid):
        x=session(m,ok,p[a:b],priors(),geometry.curve.codec())
        path=m.cumsum(0)
        ts=np.arange(40+day%5,b-a-60,5)
        for k in (0,1):
            names=replica.SIDE!=k
            live=ok[ts][:,names];w=live/live.sum(1).clip(1)[:,None]
            for j,(s,h) in enumerate(((5,20),(20,60))):
                full=(w*y[a+ts][:,names,j+1]).sum(1)
                distal=(w*(path[ts+h]-path[ts+s])[:,names]).sum(1)
                v=np.column_stack((np.ones(len(ts)),x[ts,j,k],full,distal))
                g[j]+=v.T@v
            n+=len(ts)
    return g/n


def train(end):
    rows=np.stack([moment(d) for d in range(4,end)])
    cov=rows[:,:,1:,1:]-rows[:,:,0,1:,None]*rows[:,:,None,0,1:]
    co=np.zeros((2,2,D+1))
    for j in range(2):
        ds=cov[:,j];g=ds.mean(0)
        for arm,start in enumerate((3,6)):
            ids=np.arange(start,start+3)
            proj=np.eye(D)[:,ids]
            ctrl=g[:start,:start]
            proj[:start]=-np.linalg.solve(ctrl+.04*np.diag(ctrl.diagonal().clip(1e-8)),g[:start,ids])
            sub=proj.T@g[:D,:D]@proj
            sd=sub.diagonal().clip(1e-7)**.5
            e,u=np.linalg.eigh(sub/np.outer(sd,sd))
            axis=proj@(u/(sd[:,None]*np.sqrt(e.clip(0)+.8)))
            beta=(ds[:,j+1,D]/ds[:,j+1,j+1].clip(1e-7)).clip(.2,4)
            target=ds[:,:D,D]/beta[:,None]-ds[:,:D,j+1]
            b=target@axis
            mean=b.mean(0)
            w=axis@(mean**3/(mean**2+2*b.var(0)/len(b)+1e-8))
            co[j,arm,1:]=w
            co[j,arm,0]=-rows[:,j,0,1:D+1].mean(0)@w
    return np.r_[priors().ravel(),geometry.curve.codec(),co.ravel()]


def signals(raw,grid,p,bank):
    co=bank[5007:].reshape(2,2,D+1)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x=session(m,ok,p[a:b],bank[:5000].reshape(2,50,50),bank[5000:5007])
        for j in (0,1):
            values=x[:,j]@co[j,:,1:].T+co[j,:,0]
            left,right=values[:,0].sum(1),values[:,1].sum(1)
            z=np.column_stack((values.mean(1),np.where(left*right>0,np.sign(left)*np.minimum(abs(left),abs(right)),0.)))
            out[a:b,:,:,j]=z[:,None]
    return out


def audit():
    report={}
    for end in (8,12):
        m=np.mean([moment(d) for d in range(end,end+4)],axis=0)[1]
        g=m[1:,1:]-np.outer(m[0,1:],m[0,1:])
        def summary(c):
            return (c[-2,:6]/np.sqrt(c[-2,-2]*c.diagonal()[:6]).clip(1e-12)).tolist()
        report[end+1]=dict(raw=summary(g[3:,3:]),partial=summary(g[3:,3:]-g[3:,:3]@np.linalg.solve(g[:3,:3],g[:3,3:])))
    (OUT/'issued_quote_information.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    strip.evaluation.rolling(sys.modules[__name__])
    audit()
