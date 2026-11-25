"""Separate the first price-change mark from later event marks before pricing each return horizon."""
import json
import sys

import numpy as np

import experiment_y3_cross_market_marked_renewal_v1 as parent
import y3_cross_market_renewal_references as history
import y3_four_day_neural_references as warm

OUT=parent.OUT.parent/'joint_first_event_transfer_v1'
ARMS=('unchanged','domestic accumulated value','reassigned sign prior','all transferred marks',
      'only the first mark','only subsequent marks','first mark with transferred continuation',
      'later marks with transferred start','interaction of first and later marks','remote subwindow',
      'remote first response','remote later response','replicated international direction')
M,A,D=parent.M,parent.A,parent.D


def table(rows):
    stocks,rates=parent.pace()
    opp,yes,no,positive,negative=np.moveaxis(np.sum(rows,axis=0),4,0)
    cnt=yes+no
    hazard=(cnt+60*(cnt.sum((1,2),keepdims=True)+6*rates[:,None,None,None])/
            (opp.sum((1,2),keepdims=True)+6))/(opp+60)
    def mag(amount,count):
        avg=(amount.sum((1,2),keepdims=True)+1)/(count.sum((1,2),keepdims=True)+1)
        return (amount+25*avg)/(count+25)
    plus,minus=mag(positive,yes),mag(negative,no)
    lag=np.arange(A+1)
    bins=np.stack([parent.delay(lag,r) for r in rates])
    counts=np.zeros((M,3,D,2))
    for g in range(4):
        for j in range(D):
            for k,v in enumerate((no,yes)):
                counts[:,:,j,k]+=v[g][:,:,bins[g]==j].sum(-1)
    def prob(c):
        marginal=(c.sum(2)[:,:,1]+4)/(c.sum((2,3))+8)
        return (c[:,:,:,1]+30*marginal[:,:,None])/(c.sum(3)+30)
    home=prob(counts)
    fi,us=[prob(v) for v in parent.external()]
    shared=home+.4*np.where((fi-home)*(us-home)>0,np.sign(fi-home)*np.minimum(abs(fi-home),abs(us-home)),0)
    def expand(c):
        return np.stack([c[:,:,q] for q in bins])
    h,f,s,w=[expand(v) for v in (home,.6*home+.4*fi,shared,.6*home+.4*np.roll(fi,5,axis=0))]
    # Models 4 and 5 switch their transition kernel irreversibly at the first observed price move.
    kernels=[h,f,s,w,f,h]
    after=[0,1,2,3,0,1]
    mem=np.zeros((6,*hazard.shape))
    hist={0:mem.copy()}
    m=np.arange(M)[:,None]
    b=np.arange(3)[None,:]
    for step in range(1,61):
        nxt=[]
        for i,(c,who) in enumerate(zip(kernels,after)):
            v=mem[who]
            same=v[:,(2*m+1)%M,b,0][:,:,:,None]
            reverse=v[:,(2*m)%M,2-b,0][:,:,:,None]
            nxt.append((1-hazard)*mem[i][:,:,:,np.minimum(lag+1,A)]+hazard*(c*(plus+same)-(1-c)*(minus+reverse)))
        mem=np.stack(nxt)
        if step in (5,20,60):
            hist[step]=mem.copy()
    vals=[]
    for a,b in ((0,5),(5,20),(20,60)):
        h,f,s,w,first,later=hist[b]
        prev=hist[a]
        isolated=hist[b]-prev
        first_diff=first-h
        later_diff=later-h
        vals.append(np.stack((h,w-h,f-h,first_diff,later_diff,f-later,f-first,f-h-first_diff-later_diff,
                              isolated[1]-isolated[0],isolated[4]-isolated[0],isolated[5]-isolated[0],s-h)))
    return np.asarray(vals,dtype='float32')


def signals(raw,w,stocks):
    out=np.zeros((*raw.shape[:2],3,len(ARMS)-1),np.float32)
    for lo,hi in parent.segments(len(raw)):
        for s,r in enumerate(parent.raw_returns(raw[lo:hi]).T):
            m,b,a,sign=parent.grid(r)
            vals=w[:,:,stocks[s],m,b,a]
            out[lo:hi,s]=np.moveaxis(vals,-1,0)*sign[:,None,None]
    return out-out.mean(1,keepdims=True)


def record(p,y,z):
    mats=[]
    for j in range(3):
        arr=np.concatenate((np.ones_like(p[:,:,:1]),p[:,:,j:j+1],z[:,:,j],y[:,:,j:j+1]),2)
        v=arr.reshape(-1,arr.shape[-1]).astype(float)
        mats.append(v.T@v)
    return np.stack(mats)


def scales(rows):
    cov=np.array([[[parent.covariance(m) for m in ref] for ref in day] for day in rows])
    beta=(cov[:,:,:,0,-1]/cov[:,:,:,0,0]).clip(.35,3)
    grad=(cov[:,:,:,1:-1,-1]/beta[:,:,:,None]-cov[:,:,:,1:-1,0]).mean(1)
    var=(cov.diagonal(axis1=-2,axis2=-1)[:,:,:,1:-1]-cov[:,:,:,1:-1,0]**2/cov[:,:,:,0,0,None]).mean(1)
    mean=grad.mean(0)
    return .45*mean*abs(mean)/((abs(mean)+grad.std(0)/np.sqrt(len(rows))+1.e-7)*var.mean(0).clip(1e-6))


def scores(m,weights):
    out=[]
    for row,w in zip(m,weights):
        c=parent.covariance(row)
        xy=c[0,-1]+w*c[1:-1,-1]
        vx=c[0,0]+2*w*c[0,1:-1]+w*w*c.diagonal()[1:-1]
        out.append(np.r_[0.,xy/np.sqrt(vx*c[-1,-1])-c[0,-1]/np.sqrt(c[0,0]*c[-1,-1])])
    return out


def train():
    OUT.mkdir(exist_ok=True)
    stocks,_=parent.pace()
    stats=[];rows=[];folds=[]
    for day in range(16):
        raw=parent.trial.clock.book.rawday(day)
        if day in (4,8,12):
            w=table(stats)
        if day in (8,12):
            coef=scales(rows)
            fold=[]
        if day>=4:
            z=signals(raw,w,stocks)
            if day==4:
                np.testing.assert_allclose(signals(raw[:159],w,stocks),z[:159],atol=1e-6,rtol=0)
            y=parent.trial.clock.labels(day).reshape(-1,50,3)
            refs=warm.read(day,raw)
            rec=[record(p,y,z) for p in refs]
            rows.append(rec)
            if day>=8:
                fold.append(rec)
                if day in (11,15):
                    folds.append([scores(m,coef) for m in np.sum(fold,axis=0)])
        stats.append(parent.moment(raw))
    f=np.asarray(folds)
    picks=[]
    for k in range(3):
        vs=f[:,:,k]
        good=[j for j in range(1,len(ARMS)) if j!=2 and vs[:,:,j].min()>0 and
              (vs[:,:,j].mean(0)>vs[:,:,2].mean(0)).all()]
        picks.append(max(good,key=lambda j:vs[:,:,j].min()) if good else 0)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=picks,folds=f.tolist()),indent=2))
    print('selected',[ARMS[j] for j in picks],flush=True)
    if any(picks):
        weight=scales(rows)
        coeff=np.array([weight[k,i-1] if i else 0 for k,i in enumerate(picks)])
        np.savez(OUT/'model.npz',w=table(stats),stocks=stocks,pick=np.maximum(0,np.array(picks)-1),coef=coeff)
    return any(picks)


def correction():
    with np.load(OUT/'model.npz') as f:
        w,s,i,c=f['w'],f['stocks'],f['pick'],f['coef']
    def update(raw):
        z=signals(raw,w,s)
        return (z[:,:,np.arange(3),i]*c).astype('float32')
    return update


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
    parent.replay.replay(sys.modules[__name__])
