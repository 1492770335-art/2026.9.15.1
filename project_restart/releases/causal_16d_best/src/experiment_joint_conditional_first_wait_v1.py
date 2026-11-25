"""Condition the domestic time to the next price move on observable quote activity before that move."""
import json
import sys

import numpy as np
from scipy.signal import lfilter

import joint_first_event_references as history

parent=history.incumbent
base=parent.parent
OUT=parent.OUT.parent/'joint_conditional_first_wait_v1'
M,A,D=parent.M,parent.A,parent.D
ARMS=('unchanged','unconditional domestic clock','incorrect activity','first wait with observed activity',
      'first wait with domestic marks','first wait with changed jump size','first wait against wrong stock',
      'following waits at the same pace','only subsequent waits','combined timing and size','distant interval timing')


def context(raw):
    move=(abs(raw[:,:,0].astype(float)-.00136)>.022).astype(float)
    changed=np.zeros_like(move)
    changed[1:]=np.any(raw[1:,:,[21,39,41,42]]!=raw[:-1,:,[21,39,41,42]],axis=2)
    # Compare two one-sided time scales rather than using a day's future event count.
    def average(v,h):
        rate=1-np.exp(-1/h)
        return lfilter([rate],[1,rate-1],v,axis=0,zi=(1-rate)*v[:1])[0]
    signal=.6*(average(changed,10)-average(changed,120))+.4*(average(move,12)-average(move,160))
    return np.digitize(signal,[-.065,.065])


def stats(raw):
    stocks,_=base.pace()
    out=np.zeros((4,M,3,A+1,3,5))
    for lo,hi in base.segments(len(raw)):
        x=raw[lo:hi]
        ctx=context(x)
        for stock,r in enumerate(base.raw_returns(x).T):
            motif,b,age,sign=base.grid(r)
            good=sign[:-1]!=0
            idx=(((motif[:-1]*3+b[:-1])*(A+1)+age[:-1])*3+ctx[:-1,stock])[good]
            signed=(r[1:]*sign[:-1])[good]
            yes,no=signed>base.CUT,signed < -base.CUT
            for j,w in enumerate((np.ones(len(idx)),yes,no,signed.clip(0,6)*yes,(-signed).clip(0,6)*no)):
                out[stocks[stock],:,:,:,:,j]+=np.bincount(idx,weights=w,minlength=M*3*(A+1)*3).reshape(M,3,A+1,3)
    return out


def table(records):
    pace=base.pace()[1]
    detail=np.sum(records,0)
    total=detail.sum(-2)
    opp,yes,no,m1,m0=np.moveaxis(total,-1,0)
    cnt=yes+no
    prob=(cnt+60*(cnt.sum((1,2),keepdims=True)+6*pace[:,None,None,None])/(opp.sum((1,2),keepdims=True)+6))/(opp+60)
    def size(amount,n):
        baseline=(amount.sum((1,2),keepdims=True)+1)/(n.sum((1,2),keepdims=True)+1)
        return (amount+25*baseline)/(n+25)
    pos,neg=size(m1,yes),size(m0,no)
    oc,yc,nc,mc1,mc0=np.moveaxis(detail,-1,0)
    # Shrink conditional rate ratios toward a pooled estimate at the same activity and age.
    pooled=(yc+nc).sum((1,2),keepdims=True)
    den=oc.sum((1,2),keepdims=True)
    mult=(pooled+12)/(den+60)/((cnt.sum((1,2),keepdims=True)+36)/(opp.sum((1,2),keepdims=True)+180))[:,:,:,:,None]
    prior=(prob[:,:,:,:,None]*mult.clip(.4,2.5)).clip(.002,.998)
    conditional=(yc+nc+70*prior)/(oc+70)
    pc=(mc1+40*pos[:,:,:,:,None])/(yc+40)
    rc=(mc0+40*neg[:,:,:,:,None])/(nc+40)
    lag=np.arange(A+1)
    bins=np.stack([base.delay(lag,r) for r in pace])
    ds=np.zeros((M,3,D,2))
    for g in range(4):
        for j in range(D):
            ds[:,:,j]=np.stack((no[g][:,:,bins[g]==j].sum(-1),yes[g][:,:,bins[g]==j].sum(-1)),2)+ds[:,:,j]
    def marks(c):
        prior=(c.sum(2)[:,:,1]+4)/(c.sum((2,3))+8)
        return (c[:,:,:,1]+30*prior[:,:,None])/(c.sum(3)+30)
    home=marks(ds)
    fi=.6*home+.4*marks(base.external()[0])
    home,fi=[np.stack([p[:,:,q] for q in bins]) for p in (home,fi)]
    m=np.arange(M)[:,None];b=np.arange(3)[None,:]
    out=[]
    for speed in range(3):
        hz=conditional[:,:,:,:,speed]
        ps=pc[:,:,:,:,speed];rs=rc[:,:,:,:,speed]
        # First two kernels are stationary anchors. The other kernels can change after the first event.
        profiles=((prob,home,pos,neg,0),(prob,fi,pos,neg,1),(hz,fi,pos,neg,1),(hz,home,pos,neg,0),
                  (hz,fi,ps,rs,1),(hz,fi,pos,neg,5),(prob,fi,pos,neg,5),(hz,fi,ps,rs,7))
        mem=np.zeros((len(profiles),*prob.shape))
        snapshots={0:mem.copy()}
        for k in range(1,61):
            nxt=[]
            for i,(h,c,pp,rr,post) in enumerate(profiles):
                g=mem[post]
                same=g[:,(m*2+1)%M,b,0][:,:,:,None]
                reverse=g[:,m*2%M,2-b,0][:,:,:,None]
                nxt.append((1-h)*mem[i][:,:,:,np.minimum(lag+1,A)]+h*(c*(pp+same)-(1-c)*(rr+reverse)))
            mem=np.stack(nxt)
            if k in (5,20,60):
                snapshots[k]=mem.copy()
        curves=[]
        for a,bnd in ((0,5),(5,20),(20,60)):
            v=snapshots[bnd]
            local=v[2]-v[1]
            curves.append(np.stack((v[0],local,local,v[3]-v[0],v[4]-v[2],local,v[5]-v[1],
                                    v[6]-v[1],v[7]-v[1],local-(snapshots[a][2]-snapshots[a][1]))))
        out.append(curves)
    return np.array(out,dtype='float32')


def signals(raw,w,stocks):
    out=np.zeros((*raw.shape[:2],3,len(ARMS)-1),np.float32)
    for lo,hi in base.segments(len(raw)):
        ctx=context(raw[lo:hi])
        for s,r in enumerate(base.raw_returns(raw[lo:hi]).T):
            m,b,a,sign=base.grid(r)
            g=ctx[:,s]
            vals=w[g,:,:,stocks[s],m,b,a]
            vals[:,:,1]=w[2-g,:,1,stocks[s],m,b,a]
            vals[:,:,5]-=w[ctx[:,(s+17)%50],:,5,stocks[s],m,b,a]
            out[lo:hi,s]=vals*sign[:,None,None]
    return out-out.mean(1,keepdims=True)


def train():
    OUT.mkdir(exist_ok=True)
    stocks,_=base.pace()
    states=[];rows=[];folds=[]
    for d in range(16):
        raw=history.trial.clock.book.rawday(d)
        if d in (4,8,12):
            w=table(states)
        if d in (8,12):
            coef=parent.scales(rows);group=[]
        if d>=4:
            z=signals(raw,w,stocks)
            if d==4:
                np.testing.assert_allclose(signals(raw[:184],w,stocks),z[:184],atol=1e-6,rtol=0)
            p=history.references(d,raw) if d>=8 else parent.warm.read(d,raw)
            y=history.trial.clock.labels(d).reshape(-1,50,3)
            ms=[parent.record(q,y,z) for q in p]
            rows.append(ms)
            if d>=8:
                group.append(ms)
                if d in (11,15):
                    folds.append([parent.scores(g,coef) for g in np.sum(group,0)])
        states.append(stats(raw))
    f=np.asarray(folds)
    picks=[]
    for k in range(3):
        v=f[:,:,k]
        good=[i for i in range(3,len(ARMS)) if v[:,:,i].min()>0 and (v[:,:,i].mean(0)>v[:,:,2].mean(0)).all()]
        picks.append(max(good,key=lambda j:v[:,:,j].min()) if good else 0)
    print('selected',picks,flush=True)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=picks,folds=f.tolist()),indent=2))
    if any(picks):
        idx=np.maximum(np.array(picks)-1,0)
        np.savez(OUT/'model.npz',w=table(states),stocks=stocks,idx=idx,coef=parent.scales(rows)[np.arange(3),idx]*(np.array(picks)>0))
    return any(picks)


def correction():
    with np.load(OUT/'model.npz') as f:
        w,s,idx,c=f['w'],f['stocks'],f['idx'],f['coef']
    return lambda raw:(signals(raw,w,s)[:,:,np.arange(3),idx]*c).astype('float32')


def layer():
    before,delta=parent.layer(),correction()
    return lambda raw,initial:before(raw,initial)+delta(raw)


def replay_with():
    delta=correction()
    def read(d,raw):
        p,h,n=history.read(d,raw)
        return p+delta(raw),h+delta(raw[:n]),n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
