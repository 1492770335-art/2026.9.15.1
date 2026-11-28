"""Price future book revisions through the part predicted by independent current fields."""
from functools import lru_cache
import numpy as np

import experiment_dense_hidden_refresh_interpolation_v1 as source

onset,rel,book,REFERENCE = source.onset,source.rel,source.book,source.REFERENCE
OUT=REFERENCE.parent/"dense_refresh_payoff_instruments_v1"
ARMS=("realized_revision_payoff","observable_revision_payoff","instrumented_payoff")
D=len(source.FIELDS)
C=D+3


@lru_cache(maxsize=1)
def teachers():
    rows=np.mean([np.load(source.OUT/f"gram_{d:02d}.npy")[1] for d in range(4)],axis=0)
    out=[]
    for n in (3,source.INPUTS):
        cov=rows[:,:n,:n]
        sd=cov.diagonal(axis1=1,axis2=2).clip(.001)
        out.append(np.linalg.solve(cov+.16*sd[:,:,None]*np.eye(n),rows[:,:n,-1:])[:,:,0])
    return out


def panels(raw,grid,p,stride,offset,with_future):
    base,complete=teachers()
    for a,b,v,y,last in source.sessions(raw,grid):
        total=np.concatenate((np.zeros_like(y[:1]),y.cumsum(0,dtype=float))) if with_future else None
        times=np.arange(offset,b-a-(8 if with_future else 0),stride)
        for i in range(0,len(times),64):
            t=times[i:i+64]
            x,held,age=source.rows(v,y,t,last[t])
            mask=age>=2
            passive=np.einsum("tsfd,fd->tsf",x[:,:,:,:3],base)*mask
            natural=np.einsum("tsfd,fd->tsf",x,complete)*mask
            obs=np.concatenate((passive,p[a+t],natural-passive),axis=2)
            obs-=obs.mean(1,keepdims=True)
            actual=None
            if with_future:
                actual=((total[t+9]-total[t+1])/8-held)*mask-passive
                actual-=actual.mean(1,keepdims=True)
            yield a+t,obs,actual


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(4,16):
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=(np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else
           onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        cur=p-p.mean(1,keepdims=True)
        label=y-y.mean(1,keepdims=True)
        scale=((cur*label).sum((0,1))/(cur*cur).sum((0,1))).clip(.3,3)
        err=label/scale-cur
        g=np.zeros((C+2*D+3,C+2*D+3))
        n=0
        for t,x,actual in panels(raw,grid,p,6,d%6,True):
            z=np.concatenate((x,actual,err[t]),axis=2).reshape(-1,len(g))
            g+=z.T@z;n+=len(z)
        np.save(OUT/f"gram_{d:02d}.npy",g/n)


def solve(cov,rhs,penalty=.4):
    return np.linalg.solve(cov+penalty*np.diag(cov.diagonal().clip(1e-5)),rhs)


def train(end):
    gs=np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)])
    g=gs.mean(0)
    transform=np.eye(len(g))
    transform[:C,C:C+2*D]=-solve(g[:C,:C],g[:C,C:C+2*D],.05)
    rows=transform.T@gs@transform
    X,A=slice(C,C+D),slice(C+D,C+2*D)
    w=np.zeros((D,len(ARMS),3))
    for parity in (0,1):
        first,second=[rows[k::2].mean(0) for k in (parity,1-parity)]
        stage=solve(first[X,X],first[X,A])
        w[:,0]+=.5*stage@solve(second[A,A],second[A,-3:])
        w[:,1]+=.5*solve(second[X,X],second[X,-3:])
        calibrated=solve(stage.T@second[X,X]@stage,stage.T@second[X,-3:],1.)
        w[:,2]+=.5*stage@calibrated
    return np.concatenate((np.einsum("ij,jah->iah",transform[:C,X],w),w))


def design(raw,grid,code,p,bank):
    out=np.zeros((*p.shape[:2],len(ARMS),3))
    for t,x,_ in panels(raw,grid,p,1,0,False):
        out[t]=(x@bank.reshape(C+D,-1)).reshape(len(t),50,len(ARMS),3)
    return out


def predict(raw,grid,p,params):
    return book.path.corrected(p,design(raw,grid,params["code"],p,params["bank"]),params)


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
