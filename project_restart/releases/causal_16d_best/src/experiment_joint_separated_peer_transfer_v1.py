"""Separate the market from relative forecasts before estimating peer transfer directions."""
import json
import sys
import numpy as np
import torch

import experiment_joint_peer_transfer_operator_v1 as peer

history, parent, trial, stack = peer.history, peer.parent, peer.trial, peer.stack
OUT = trial.clock.ART / 'joint_separated_peer_transfer_v1'
ARMS = ('unchanged','mixed transfer','market predicts relative returns','relative forecasts predict market',
        'relative to relative','transfers involving market','separately fitted transfers',
        'rotated recipient identities','between relative directions with rank two','repeated market transfers')


def gradients(records,k):
    base = peer.unit(k)
    gs,bs=[],[]
    for m in np.mean(records,axis=1):
        n,_,_,g,xy,_=peer.terms(m,k)
        beta=np.clip(peer.correlation(m,base,k,True),.35,3.)
        gs.append(g/n)
        bs.append((xy/beta-g@base)/n)
    return np.mean(gs,0),np.asarray(bs)


def readouts(records,k):
    mixed=peer.readouts(records,k)[4]
    g,grad=gradients(records,k)
    def fit(i,j,rank=6):
        scale=g.diagonal()[i].clip(.00001)**.5
        z=grad[:,i][:,:,j]/scale[None,:,None]
        mask=i[:,None]//3!=j[None,:]
        mu=z.mean(0)
        shrink=abs(mu)/(abs(mu)+z.std(0)/np.sqrt(len(z))+1e-7)
        u,s,v=np.linalg.svd(mu*shrink*mask,full_matrices=False)
        u,s,v=u[:,:rank],s[:rank],v[:rank]
        signs=np.einsum('ia,dij,aj->da',u,z*mask,v,optimize=True)
        n=len(records)//2
        labs=[signs[:n],signs[n:],signs[::2],signs[1::2]]
        low=np.min([item.mean(0) for item in labs],axis=0).clip(0)
        stable=(u*np.minimum(low,s))@v
        cov=g[np.ix_(i,i)]/scale[:,None]/scale[None,:]
        w=.35*np.linalg.solve(cov+1.8*np.eye(len(i)),stable)/scale[:,None]
        result=np.zeros_like(mixed)
        result[np.ix_(i,j)]=w*mask
        return result
    m,r=np.arange(3),np.arange(3,150)
    market,rest=np.arange(1),np.arange(1,50)
    mr=fit(m,rest)
    rm=fit(r,market)
    rr=fit(r,rest)
    u=peer.U()[0].astype(float)
    perm=u.T@np.roll(u,17,axis=0)
    all_terms=mr+rm+rr
    fake=np.einsum('ij,jhk,lk->ihl',perm,all_terms.reshape(50,3,50),perm).reshape(150,50)
    # Keep market-linked coefficients when the joint and the restricted fits agree in sign.
    both=mr+rm
    repeated=np.sign(both)*np.minimum(abs(both),abs(mixed))*(both*mixed>0)
    return np.array([np.zeros_like(mixed),mixed,mr,rm,rr,mr+rm,all_terms,fake,fit(r,rest,2),repeated])


def audit():
    records=np.asarray([peer.record(d) for d in range(8,16)])
    folds=[]
    pieces=('full','market to relative','relative to market','relative to relative')
    for n in (2,4,6):
        w=peer.readouts(records[:n],2)[4]
        m=np.zeros_like(w);m[:3,1:]=w[:3,1:]
        p=np.zeros_like(w);p[3:,0]=w[3:,0]
        r=w-m-p
        weights=np.stack((np.zeros_like(w),w,m,p,r))
        folds.extend([[peer.gains(gram,weights,2)[1:].tolist() for gram in d] for d in records[n:n+2]])
    vals=np.array(folds)
    OUT.mkdir(exist_ok=True)
    (OUT/'transfer_attribution.json').write_text(json.dumps(dict(parts=pieces,days=list(range(11,17)),daily=folds,
                mean=vals.mean(0).tolist(),positive_days=(vals>0).sum(0).tolist()),indent=2))
    print('daily Y3 transfer anatomy',np.round(vals[:,0],5),flush=True)


def adjustment():
    return peer.adjustment(OUT)


def layer():
    before,adjust=parent.layer(),adjustment()
    return lambda raw,base: adjust(before(raw,base))


def replay_with():
    adjust=adjustment()
    def read(d,raw):
        p,h,n=history.read(d,raw)
        return adjust(p),adjust(h),n
    return stack.verifier,read


if __name__=='__main__':
    torch.set_num_threads(2)
    mod=sys.modules[__name__]
    if sys.argv[1:]==['replay']:
        peer.replay.replay(mod)
    else:
        audit()
        if peer.train(mod):
            peer.replay.replay(mod)
