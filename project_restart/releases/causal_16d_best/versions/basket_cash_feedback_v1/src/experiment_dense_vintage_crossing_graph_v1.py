"""Infer per-stock corrections from crossings of fixed-deadline forecasts between closely matched stocks."""
import sys

import numpy as np
import experiment_dense_terminal_vintage_ladder_v1 as ladder

strip,rel,onset,REFERENCE,read_p=ladder.strip,ladder.rel,ladder.onset,ladder.REFERENCE,ladder.read_p
OUT=REFERENCE.parent/'dense_vintage_crossing_graph_v1'
ARMS=('crossing_adjustment','crossings_above_motion','reconciled_local_edges')
K=4
C=5


def edge_blocks(raw,grid,p,unit,stride,phase,tail=0):
    for a,b,z,gaps in ladder.panels(raw,grid,p,unit):
        ticks=np.arange(55+phase,b-a-tail,stride)
        for offset in range(0,len(ticks),48):
            t=ticks[offset:offset+48]
            q=z[t,:,1:4]
            price5=z[t,:,1]-z[t,:,4]
            price20=z[t,:,2]-z[t,:,13]
            lagged=np.stack((price5,price20),axis=2)
            ix=np.arange(len(t))[:,None,None]
            profile=(q-q.mean(1,keepdims=True))/(.09+q.std(1,keepdims=True))
            delta=profile[:,:,None]-profile[:,None]
            dist=(delta*delta).sum(3)
            ok=gaps[t+1]==gaps[t-55]
            dist+=100*(~ok[:,None,:])
            dist[:,np.arange(50),np.arange(50)]=np.inf
            peers=np.argpartition(dist,K-1,axis=2)[:,:,:K]
            wt=np.exp(-.4*np.take_along_axis(dist,peers,axis=2))*ok[:,:,None]*ok[ix,peers]

            def difference(v):
                return v[:,:,None]-v[ix,peers]
            now=difference(q)
            then=difference(lagged)
            # Convert all columns to signed pair contrasts: reversing the two issuers
            # reverses the suggested payoff, while leaving the edge weight unchanged.
            step=now[...,:2]-then
            gate=.5-.5*np.tanh(now[...,:2]/.06)*np.tanh(then/.09)
            broke=np.stack((z[t,:,8],z[t,:,17]),axis=-1)
            away=difference(broke)
            scale=abs(now[...,:2])+abs(then)+.06
            move=step/scale
            inputs=np.concatenate((now,step,step*gate,move*gate,away*gate,
                                   step*gate[...,::-1],move*abs(now[...,2:3]),away*np.tanh(abs(then)/.08)),axis=-1)
            yield a+t,peers,wt,inputs


def collect(end):
    unit=ladder.base_scale(end)
    _,coef,_=strip.physical()
    unit=unit*coef[2]/coef
    rows=[]
    for d in range(4,end):
        raw,grid,y=[onset.settled.load(d,k) for k in ('x','time','y')]
        p=read_p(d)
        scale=strip.daily_scale(p,y)[1:]
        err=y[...,1:].astype(float)/scale-p[...,1:]
        total=count=0
        for t,peers,w,x in edge_blocks(raw,grid,p,unit,5,d%5,tail=60):
            ix=np.arange(len(t))[:,None,None]
            e=err[t]
            dy=e[:,:,None]-e[ix,peers]
            v=np.concatenate((x,dy),axis=-1)
            row=v.reshape(-1,v.shape[-1])
            total+=row.T@(w.reshape(-1,1)*row)
            count+=w.sum()
        rows.append(total/count)
    return unit,np.stack(rows)


def train(end):
    unit,days=collect(end)
    g=days.mean(0)
    n=len(g)-2
    v=np.eye(n)[:,C:]
    v[:C]=-np.linalg.solve(g[:C,:C]+.008*np.diag(g[:C,:C].diagonal().clip(1e-7)),g[:C,C:n])
    weights=[]
    for proj in (np.eye(n)[:,C:],v):
        cov=proj.T@g[:n,:n]@proj
        pay=proj.T@days[:,:n,n:]
        avg=pay.mean(0)
        reg=avg**3/(avg*avg+pay.var(0)/len(days)+1e-10)
        co=np.linalg.solve(cov+.45*np.diag(cov.diagonal().clip(1e-7))+1e-8*np.eye(len(cov)),reg)
        weights.append(proj@co)
    return np.r_[unit,np.stack(weights).ravel()]


def recover(peers,weight,edge):
    size=len(peers)
    ix=np.arange(size)[:,None,None]
    ii=np.arange(50)[None,:,None]
    lap=np.broadcast_to(np.eye(50),(size,50,50)).copy()
    rhs=np.zeros((size,50,edge.shape[-1]))
    # A directed observation contributes one undirected squared difference loss.
    weighted=weight[...,None]*edge
    rhs+=weighted.sum(2)
    np.add.at(rhs,(ix,peers),-weighted)
    lap[:,np.arange(50),np.arange(50)]+=weight.sum(2)
    np.add.at(lap,(ix,peers,peers),weight)
    np.add.at(lap,(ix,ii,peers),-weight)
    np.add.at(lap,(ix,peers,ii),-weight)
    return np.linalg.solve(lap,rhs)


def signals(raw,grid,p,bank):
    out=np.zeros((*p.shape[:2],3,2))
    w=bank[3:].reshape(2,-1,2)
    for t,peers,wt,x in edge_blocks(raw,grid,p,bank[:3],1,0):
        for j in range(2):
            edge=x@w[j]
            if j==1:
                num=(wt[...,None]*edge).sum(2)
                out[t,:,j]=strip.center(num/(1+wt.sum(2))[...,None])
                out[t,:,2]=recover(peers,wt,edge)
            else:
                out[t,:,j]=recover(peers,wt,edge)
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    strip.evaluation.rolling(sys.modules[__name__])
