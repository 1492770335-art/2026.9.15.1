"""Estimate odd nonlinear book responses and their change when the long forecast revises the near direction."""
from functools import lru_cache
import sys
import numpy as np
import torch
from scipy.signal import lfilter

from experiment_dense_clock_coincidence_risk_v1 import REFERENCE,read_p,event,onset,rel,strip

OUT=REFERENCE.parent/'dense_conditional_book_kernel_v1'
ARMS=('book_kernel','long_disagreement_kernel','shared_across_dates')
WIDTH=64
SIGNED=14
EVEN=8
CTRL=SIGNED+3
D=CTRL+2*WIDTH
rng=np.random.default_rng(628133)
S=rng.normal(size=(SIGNED,WIDTH))/np.sqrt(SIGNED)*1.6
E=rng.normal(size=(EVEN,WIDTH))/np.sqrt(EVEN)*1.5
PHASE=rng.uniform(0,2*np.pi,size=WIDTH)


def smooth(x,tau):
    decay=np.exp(-1/tau)
    return x[:1]+lfilter([1-decay],[1.,-decay],x-x[:1],axis=0)


def views(raw,m,p):
    fields=[41,42,43,44,50,51,54,55,77,78]
    a=np.tanh(.7*np.arcsinh(raw[:,:,fields].astype(float)))
    side=a[...,::2]-a[...,1::2]
    even=a[...,::2]+a[...,1::2]
    fast,slow=smooth(side,5),smooth(side,45)
    q=strip.center(p.astype(float))
    std=np.sqrt((q*q).mean(1,keepdims=True)).clip(.015)
    z=np.tanh(q/std)
    vol=np.sqrt(smooth(m*m,45).clip(.0001))
    ret=np.tanh(smooth(m,12)/(.01+vol)*3)
    signed=np.concatenate((side,fast-slow,ret[...,None],z),axis=2)
    orient=np.tanh((z[:,:,2]-z[:,:,1])*2)*z[:,:,1]
    quiet=np.tanh(np.log(vol/.08))
    neutral=np.concatenate((even,quiet[...,None],orient[...,None],abs(z[:,:,2:3])),axis=2)
    return q,signed,neutral,orient


def design(view,t):
    q,signed,neutral,orient=[v[t] for v in view]
    odd=np.sin(signed@S)*np.cos(neutral@E+PHASE)
    return strip.center(np.concatenate((q,signed,odd,odd*orient[...,None]),axis=2))


@lru_cache(maxsize=12)
def record(day):
    p=read_p(day)
    raw,grid,y=[onset.settled.load(day,k) for k in ('x','time','y')]
    slope=strip.daily_scale(p,y).clip(.2,4.)
    _,coef,_=strip.physical()
    gram=np.zeros((D+3,D+3))
    count=0
    for a,b,m,ok in event.sessions(raw,grid):
        v=views(raw[a:b],m,p[a:b])
        times=np.arange(40+day%6,b-a-60,6)
        for lo in range(0,len(times),96):
            t=times[lo:lo+96]
            x=design(v,t)
            error=strip.center(y[a+t].astype(float)/slope-p[a+t])
            far=error[:,:,2]-error[:,:,1]*slope[1]/slope[2]*coef[2]/coef[1]
            r=np.concatenate((x,error[:,:,1:],far[...,None]),axis=2)
            r=r[ok[t]]
            gram+=r.T@r
            count+=len(r)
    return gram/count


def maximin(rows):
    n=len(rows)
    G=rows@rows.T
    # Shrink the worst-date optimization towards equal date weights.
    G+=np.eye(n)*max(float(np.trace(G)/n),1e-8)
    w=np.ones(n)/n
    step=1/max(np.linalg.norm(G,2),1e-8)
    for _ in range(80):
        v=w-step*(G@w)
        order=np.sort(v)[::-1]
        offsets=(order.cumsum()-1)/np.arange(1,n+1)
        rho=np.flatnonzero(order>offsets)[-1]
        w=np.maximum(v-offsets[rho],0.)
    return w@rows


def train(end):
    samples=np.stack([record(d) for d in range(4,end)])
    mean=samples.mean(0)
    ws=[]
    for mode in range(3):
        start=CTRL if mode==0 else CTRL+WIDTH
        ids=np.arange(start,start+WIDTH)
        g=mean[:start,:start]
        v=np.eye(D)[:,ids]
        v[:start]=-np.linalg.solve(g+.11*np.diag(g.diagonal().clip(1e-6)),mean[:start,ids])
        gram=v.T@mean[:D,:D]@v
        sd=gram.diagonal().clip(1e-7)**.5
        eig,axis=np.linalg.eigh(gram/np.outer(sd,sd))
        axes=(axis/np.sqrt(eig.clip(0)+1.5))/sd[:,None]
        grad=(v@axes).T@samples[:,:D,D:]
        m=grad.mean(0)
        safe=m*m/(m*m+2*grad.var(0)/len(samples)+1e-10)
        safe[:,1]*=m[:,1]*m[:,2]>0
        if mode==2:
            m=np.stack([maximin(grad[:,:,k]) for k in range(3)],axis=1)
        ws.append(v@axes@(m*safe)[:,:2])
    return np.stack(ws)


def signals(raw,grid,p,bank):
    out=np.zeros((*p.shape[:2],3,2))
    for a,b,m,ok in event.sessions(raw,grid):
        v=views(raw[a:b],m,p[a:b])
        for lo in range(0,b-a,64):
            t=np.arange(lo,min(lo+64,b-a))
            x=design(v,t)
            out[a+t]=np.einsum('tsx,jxh->tsjh',x,bank,optimize=True)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    strip.evaluation.rolling(sys.modules[__name__])
