"""Predict endpoint returns by propagating causal decaying order-flow traces and accumulating short rewards."""
import sys
import numpy as np
from scipy.signal import lfilter

import experiment_dense_forward_strip_observer_v1 as strip

base,rel,onset,REFERENCE=strip.base,strip.rel,strip.onset,strip.REFERENCE
OUT=REFERENCE.parent/"dense_trace_reward_rollout_v1"
ARMS=("short_reward_rollout","short_then_medium_reward","known_flow_tail")
POLES=np.array([0.,.56,.86,.965])
CHANNELS=16
WIDTH=CHANNELS*len(POLES)
HORIZONS=(5,20,60)


def sessions(raw,grid):
    for a,b in rel.fields.parent.sessions(grid):
        v=raw[a:b]
        def field(ids):
            return np.tanh(.65*np.arcsinh(v[:,:,ids].astype(float)))
        book=np.concatenate((field([41,50,54,77])-field([42,51,55,78]),field([0,11,26,64])),axis=-1)
        fast=base.panel.smooth(book,7)
        slow=base.panel.smooth(book,55)
        impulses=np.concatenate((book,(fast-slow)*np.tanh(3*abs(slow))),axis=-1)
        impulses=strip.center(impulses)
        traces=[]
        for pole in POLES:
            if pole==0:
                traces.append(impulses)
            else:
                traces.append(lfilter([1-pole],[1,0,0,0,0,-pole],impulses,axis=0))
        yield a,b,np.concatenate(traces,axis=2)


def collect():
    OUT.mkdir(exist_ok=True)
    _,coef,bad=strip.physical()
    for d in range(16):
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=base.read_p(d) if d>=4 else None
        moment=np.zeros((WIDTH,WIDTH+CHANNELS+3))
        readout=np.zeros((WIDTH+3+2,WIDTH+3+2))
        n=0
        for a,b,x in sessions(raw,grid):
            t=np.arange(d%5,b-a-60,5)
            gaps=np.vstack((np.zeros(50,int),np.all(raw[a:b]==bad,axis=2).cumsum(0)))
            valid=gaps[t+61]==gaps[t]
            truth=strip.center(y[a+t].astype(float))
            xx=x[t][valid]
            nxt=x[t+5,:,:CHANNELS][valid]
            r=truth[valid]*coef[2]/coef
            moment+=xx.T@np.column_stack((xx,nxt,r))
            if p is not None:
                q=strip.center(p[a+t].astype(float))[valid]
                z=np.column_stack((xx,q,truth[valid,1:]))
                readout+=z.T@z
            n+=len(xx)
        np.savez(OUT/f"day_{d:02d}.npz",mom=moment/n,readout=readout/n)


def ridge(g,r,penalty):
    return np.linalg.solve(g+penalty*np.diag(g.diagonal().clip(1e-7))+np.eye(len(g))*1e-8,r)


def transition(m):
    A=ridge(m[:,:WIDTH],m[:,WIDTH:WIDTH+CHANNELS],.12)
    frozen=np.diag(np.repeat(POLES,CHANNELS))
    return frozen+np.concatenate([(1-v)*A for v in POLES],axis=1), frozen


def laws(rows):
    out=0.
    for j in (0,1):
        dyn,rew=[rows[k::2].mean(0) for k in (j,1-j)]
        live,held=transition(dyn)
        short,med,_=ridge(rew[:,:WIDTH],rew[:,-3:],.16).T
        result=[]
        for n in (4,12):
            all_short=sum(np.linalg.matrix_power(live,k)@short for k in range(n))
            mixed=sum(np.linalg.matrix_power(live,k)@med for k in range(0,n,4)) if n==12 else med
            known=sum(np.linalg.matrix_power(held,k)@short for k in range(n))
            result.append(np.stack((all_short,mixed,known),axis=1))
        out+=np.stack(result,axis=-1)/2
    return out


def train(end):
    mats=[];read=[]
    for d in range(end):
        with np.load(OUT/f"day_{d:02d}.npz") as f:
            mats.append(f["mom"])
            read.append(f["readout"])
    mats=np.stack(mats)
    calibration=[]
    # Each calibration date sees a reward/transition model fitted to other dates.
    for d in range(4,end):
        co=laws(np.delete(mats,d,axis=0))
        moment=read[d]
        record=[]
        for h in range(2):
            xx=np.zeros((WIDTH+5,6))
            xx[WIDTH:WIDTH+3,:3]=np.eye(3)
            xx[:WIDTH,3:]=co[:,:,h]
            target=WIDTH+3+h
            q=WIDTH+1+h
            scale=moment[q,target]/max(moment[q,q],.001)
            record.append((xx.T@moment@xx,xx.T@(moment[:,target]/max(.1,scale)-moment[:,q])))
        calibration.append(record)
    co=laws(mats)
    weights=np.zeros((WIDTH+3,3,2))
    for h in range(2):
        g,rhs=[np.stack(z) for z in zip(*(z[h] for z in calibration))]
        gmean=g.mean(0)
        proj=np.vstack((-ridge(gmean[:3,:3],gmean[:3,3:],.03),np.eye(3)))
        response=rhs@proj
        m=response.mean(0)
        w=m**3/(m**2+response.var(0)/len(rhs)+1e-9)/((proj.T@gmean@proj).diagonal()*1.5+1e-7)
        weights[:WIDTH,:,h]=co[:,:,h]*w
        weights[WIDTH:,:,h]=proj[:3]*w
    return weights


def signals(raw,grid,p,bank):
    out=np.empty((*p.shape[:2],3,2))
    for a,b,x in sessions(raw,grid):
        q=strip.center(p[a:b].astype(float))
        z=np.concatenate((x,q),axis=-1)
        out[a:b]=(z@bank.reshape(WIDTH+3,6)).reshape(b-a,50,3,2)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=="__main__":
    collect()
    strip.evaluation.rolling(sys.modules[__name__])
