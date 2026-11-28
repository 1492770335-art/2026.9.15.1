"""Cross-fit a long-short return surprise as a training control variate for the other panel's mean return."""
from functools import lru_cache
import json
import sys
import numpy as np

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip
import experiment_dense_replica_quote_updates_v1 as src

ref,REFERENCE,read_p,TARGETS=src.ref,src.REFERENCE,src.read_p,src.TARGETS
OUT=REFERENCE.parent/'dense_replica_innovation_labels_v1'
ARMS=('hedge_error_removed','two_tracking_errors_removed','replicated_clean_gradient')
D=src.D
SH=D+5


@lru_cache(maxsize=12)
def record(d):
    raw,grid,y=[onset.settled.load(d,k) for k in ('x','time','y')]
    p=read_p(d)
    g=np.zeros((2,SH,SH));mass=0
    code=src.geometry.curve.codec()
    for a,b,m,ok in event.sessions(raw,grid):
        x=src.session(m,ok,p[a:b],src.priors(),code)
        t=np.arange(40+d%5,b-a-60,5)
        for j in (0,1):
            for k in (0,1):
                names=src.replica.SIDE==k
                w=src.hedge(src.priors()[j],k)
                y0=y[a+t,:,j+1].astype(float)
                p0=p[a+t,:,j+1].astype(float)
                # Labels: opposite-panel market and issuing-panel zero-cost spread and market.
                rows=np.column_stack((np.ones(len(t)),x[t,j,k],y0[:,~names].mean(1),y0[:,names]@w,
                                      y0[:,names].mean(1),p0[:,names]@w))
                g[j]+=rows.T@rows
            if j==0:mass+=2*len(t)
    return g/mass


def residuals(g,j):
    eye=np.eye(SH-1)
    control=np.stack((eye[D+1],eye[D+2]),axis=1)
    control[:,0]-=eye[D+3]*np.clip(g[D+3,D+1]/max(g[D+3,D+3],1e-8),.1,5.)
    # Estimate the issuing panel's common price from its observable curve counterpart.
    control[:,1]-=eye[j+1]*np.clip(g[j+1,D+2]/max(g[j+1,j+1],1e-8),.1,5.)
    target=eye[D]/np.clip(g[j+1,D]/max(g[j+1,j+1],1e-8),.25,4.)-eye[j+1]
    return target,control


def cov(r):
    return r[...,1:,1:]-r[...,0,1:,None]*r[...,None,0,1:]


def train(end):
    stats=np.stack([record(d) for d in range(4,end)])
    co=np.zeros((2,3,D+1))
    for j in (0,1):
        ds=cov(stats[:,j]);g=ds.mean(0)
        proj=np.eye(D)[:,3:]
        ctrl=g[:3,:3]
        proj[:3]=-np.linalg.solve(ctrl+.04*np.diag(ctrl.diagonal().clip(1e-8)),g[:3,3:D])
        sub=proj.T@g[:D,:D]@proj
        sd=sub.diagonal().clip(1e-8)**.5
        e,u=np.linalg.eigh(sub/np.outer(sd,sd))
        axes=proj@(u/(sd[:,None]*np.sqrt(e.clip(0)+.9)))
        estimates=[]
        for parity in (0,1):
            prior=ds[parity::2].mean(0)
            target,c=residuals(prior,j)
            group=[]
            for dim in (1,2):
                cv=c[:,:dim]
                rc=cv.T@prior@cv
                weight=np.linalg.solve(rc+.15*np.diag(rc.diagonal().clip(1e-7)),cv.T@prior@target)
                clean=target-cv@weight
                group.append((ds[1-parity::2,:D]@clean)@axes)
            estimates.append(group)
        for arm in range(3):
            parts=[group[min(arm,1)] for group in estimates]
            v=np.concatenate(parts)
            mu=v.mean(0);w=mu**3/(mu**2+2*v.var(0)/len(v)+1e-8)
            if arm==2:
                w*=parts[0].mean(0)*parts[1].mean(0)>0
            co[j,arm,1:]=axes@w
            co[j,arm,0]=-stats[:,j,0,1:D+1].mean(0)@co[j,arm,1:]
    return np.r_[src.priors().ravel(),src.geometry.curve.codec(),co.ravel()]


def signals(raw,grid,p,bank):
    co=bank[5007:].reshape(2,3,D+1)
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        x=src.session(m,ok,p[a:b],bank[:5000].reshape(2,50,50),bank[5000:5007])
        for j in (0,1):
            z=x[:,j].mean(1)@co[j,:,1:].T+co[j,:,0]
            out[a:b,:,:,j]=z[:,None]
    return out


def audit():
    report={}
    for end in (8,12):
        before=cov(np.mean([record(d) for d in range(4,end)],axis=0)[1])
        after=cov(np.mean([record(d) for d in range(end,end+4)],axis=0)[1])
        eye=np.eye(SH-1)
        # The two readouts are fit only on earlier dates and held fixed on the examined dates.
        idx=[0,1,2,3,4,5,D+3]
        xx=eye[:,idx]
        var=xx.T@before@xx
        preds=xx@np.linalg.solve(var+.15*np.diag(var.diagonal().clip(1e-7)),xx.T@before[:,D:D+2])
        mh=np.eye(SH-1)[:,D:D+2];err=mh-preds
        covs=np.array([[p[:,0]@after@q[:,1] for q in (preds,err)] for p in (preds,err)])
        actual=after[D,D+1]
        target,c=residuals(before,1)
        rc=c.T@before@c
        co=np.linalg.solve(rc+.15*np.diag(rc.diagonal().clip(1e-7)),c.T@before@target)
        clean=target-c@co
        report[end+1]=dict(share_of_observed_covariance=(covs/actual).tolist(),
                            after_control_variance_ratio=float((clean@after@clean)/(target@after@target)))
    (OUT/'tracking_surprise_decomposition.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    strip.evaluation.rolling(sys.modules[__name__]);audit()
