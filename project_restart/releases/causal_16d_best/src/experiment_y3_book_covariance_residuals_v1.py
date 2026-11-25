"""Separate residual co-movement from the nonlinear product of the conditional field means."""
from functools import lru_cache
import sys
import numpy as np
import torch
import experiment_y3_future_book_comoments_v1 as base
import y3_future_book_comoment_references as history

parent,trial,stack=history.incumbent,history.trial,history.stack
OUT=trial.clock.ART/'y3_book_covariance_residuals_v1'
ARMS=('unchanged','total covariance','projected residual covariance','nonlinear mean product alone',
      'mean forecast errors','residual covariance and nonlinear mean product','cross-fitted residual covariance',
      'shuffled residual pairs','three independently priced components','cross-fitted covariance and mean errors',
      'cross-fitted covariance after subtracting shuffled pairs')
GROUPS=((),(0,),(1,),(2,),(3,),(1,2),(4,),(5,),(1,2,3),(4,6),(7,))
CANDIDATES=(2,3,4,5,6,8,9,10)
CONTROLS=(1,7)


@lru_cache(maxsize=16)
def sample(d):
    raw=trial.clock.book.rawday(d)
    n=len(raw)
    t=np.arange(90+d%13,n-60,13)
    t=t[(t+60<2401)|(t>=2491)]
    x,z,_=base.observations(raw,t,True)
    return x.reshape(-1,x.shape[-1]),z[:,:,:16].reshape(-1,16).copy()


def contact(a,b,pay):
    return ((a@pay)*b).sum(-1)


def params(end):
    fit=base.params(end)
    means=fit['full_state'][:,:16]
    pay=fit['full_pay'][16:].reshape(8,8)
    stats=[];grams=[]
    for day in range(end):
        x,z=sample(day)
        z0,z1=z[:,:8],z[:,8:]
        def parts(mu):
            a,b=mu[:,:8],mu[:,8:]
            da,db=z0-a,z1-b
            return (contact(da,db,pay),contact(a,b,pay),contact(a,db,pay)+contact(da,b,pay),
                    contact(da,np.roll(db.reshape(-1,50,8),17,axis=1).reshape(-1,8),pay))
        main=parts(x@means)
        alt=parts(x@fit[('odd' if day%2==0 else 'even')+'_state'][:,:16])
        target=np.stack((main[0],main[1],main[2],alt[0],alt[3],alt[2]),axis=1).astype(float)
        stats.append(x.T@target/len(x));grams.append(base.moments(day)[0])
    read=base.ridge(np.mean(grams,0),np.mean(stats,0))
    total=fit['full_state'][:,16:]@pay.ravel()
    # Ridge is linear in the outcome; the split must reconstruct the existing forecast exactly.
    np.testing.assert_allclose(read[:,:3].sum(1),total,rtol=1e-4,atol=2.e-6)
    return dict(means=means,pay=pay,read=read,total=total)


def advice(raw,t,st):
    x=base.observations(raw,t)[0]
    f=x@st['means']
    product=contact(f[:,:,:8],f[:,:,8:],st['pay'])
    v=x@st['read']
    total=x@st['total']-product
    curvature=v[:,:,1]-product
    cols=(total,v[:,:,0],curvature,v[:,:,2],v[:,:,3],v[:,:,4],v[:,:,5],v[:,:,3]-v[:,:,4])
    return trial.clock.center(np.stack(cols,axis=2))


def correction():
    with np.load(OUT/'model.npz') as f: st=dict(f)
    def apply(raw,p):
        out=[]
        for t in np.array_split(np.arange(len(raw)),max(1,(len(raw)+799)//800)):
            out.append(p[t]@st['w'][:3]+advice(raw,t,st)@st['w'][3:])
        return np.concatenate(out)
    return apply


def layer():
    return history.study.layer(sys.modules[__name__],parent)


def replay_with():
    change=correction()
    def read(d,raw):
        p,h,n=history.read(d,raw)
        q=p.copy();part=h.copy()
        q[:,:,2]+=change(raw,p);part[:,:,2]+=change(raw[:n],h)
        return q,part,n
    return stack.verifier,read


if __name__=='__main__':
    torch.set_num_threads(2)
    mod=sys.modules[__name__]
    if sys.argv[1:]==['replay'] or base.train(mod):
        base.model.replay.replay(mod)
