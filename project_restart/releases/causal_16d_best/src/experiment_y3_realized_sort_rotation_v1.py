"""Predict rotation between characteristic portfolios from fully settled shorter portfolio forecasts."""
import argparse
import json

import numpy as np

import experiment_compounded_return_audit_v1 as tape
import experiment_y3_session_characteristic_portfolios_v1 as portfolios
from y3_marginal_replay_v1 import evaluate

clock, parent = portfolios.clock, portfolios.parent
OUT=parent.ART/'y3_realized_sort_rotation_v1'
ARMS=('incumbent','wrong constituents','own recent return','transfer from other sorts',
      'own and other sorts','short-window transfer only','direction agreed across days','recent minus older response')
BLOCK=48


def geometry():
    with np.load(tape.OUT/'model.npz') as obj:
        return float(obj['k']),obj['coeff'],obj['row']


def states(raw,p,geo):
    k,coef,row=geo
    f=portfolios.weights(raw,p)
    price=tape.compound(raw,k)
    missing=np.vstack((np.zeros((1,50),int),(~tape.present(raw,row)).cumsum(0)))
    starts=np.arange(0,len(p),BLOCK)
    feat=[]
    for idx,h in enumerate((5,20,60)):
        t=np.arange(0,max(0,len(p)-h),4)
        if len(t):
            ok=missing[t+h+1]==missing[t]
            ret=(coef[idx,0]+coef[idx,1]*tape.span(price,t,h,k)-p[t,:,idx])*coef[2,1]/coef[idx,1]
            # Remove a common return using only stocks observed throughout this interval.
            ret-= (ret*ok).sum(1,keepdims=True)/ok.sum(1,keepdims=True).clip(1)
            fv=f[t]*ok[:,:,None]
            g=fv.transpose(0,2,1)@fv/50
            rhs=(fv*ret[:,:,None]).mean(1)
            vals=np.linalg.solve(g+.30*np.eye(5),rhs[:,:,None])[:,:,0]
            vals*= (ok.sum(1)>=25)[:,None]
            total=np.vstack((np.zeros((1,5)),vals.cumsum(0)))
        else:
            total=np.zeros((1,5))
        for left,right in ((0,144),(144,576)):
            upper=np.searchsorted(t+h,starts-left)
            lower=np.searchsorted(t+h,starts-right)
            # The additional mass shrinks very early or sparsely observed estimates toward zero.
            feat.append((total[upper]-total[lower])/(upper-lower+14)[:,None])
    return f,np.stack(feat,axis=1)


def record(raw,p,y,geo,d):
    a=parent.exposure.bands(p);b=parent.exposure.bands(y)
    u=a-a.mean((0,1));v=b-b.mean((0,1))
    unit=((u*v).sum((0,1))/(u*u).sum((0,1)).clip(.001)).clip(.45,2.5)
    err=(b/unit-a)@np.array([.62,.38])
    out=[]
    for lo,hi in parent.base.segments(len(p)):
        f,x=states(raw[lo:hi],p[lo:hi],geo)
        t=np.arange(BLOCK+d%7,len(f)-60,7)
        ft=f[t].transpose(0,2,1)
        premium=np.linalg.solve(ft@f[t]/50+.30*np.eye(5),ft@err[lo+t,:,None]/50)[:,:,0]
        z=x[t//BLOCK].reshape(len(t),-1)
        out.append((z.T@z/len(z),z.T@premium/len(z)))
    return out


def fit(hist):
    state={}
    for side,days in enumerate(zip(*hist)):
        gs,bs=[np.stack(k) for k in zip(*days)]
        def solve(g,c,keep=None):
            w=np.zeros_like(c)
            for i in range(5):
                indices=np.arange(30)
                if keep is not None:
                    indices=indices[keep(indices,i)]
                cov=g[np.ix_(indices,indices)]
                reg=1.4*np.diag(cov.diagonal().clip(.00006))
                w[indices,i]=np.linalg.solve(cov+reg,c[indices,i])
            return w
        g=gs.mean(0)
        def rhs(b):
            c=b.mean(0)
            return c*abs(c)/(abs(c)+1.3*b.std(0)/np.sqrt(len(b))+.000015)
        c=rhs(bs)
        own=solve(g,c,lambda ids,i:ids%5==i)
        other=solve(g,c,lambda ids,i:ids%5!=i)
        full=solve(g,c)
        short=solve(g,c,lambda ids,i:(ids//10<=1)&(ids%5!=i))
        common=np.zeros_like(full)
        if len(gs)>=2:
            wa,wb=[solve(gs[k::2].mean(0),rhs(bs[k::2])) for k in (0,1)]
            common=np.where(wa*wb>0,np.sign(wa)*np.minimum(abs(wa),abs(wb)),0)
        recent=solve(g,c,lambda ids,i:ids//5%2==0)
        old=solve(g,c,lambda ids,i:ids//5%2==1)
        state[f'w{side}']=np.stack((full,own,other,full,short,common,recent-old),2)
    return state


def signals(raw,p,st,geo):
    out=[]
    for j,(lo,hi) in enumerate(parent.base.segments(len(raw))):
        f,x=states(raw[lo:hi],p[lo:hi],geo)
        v=np.einsum('bc,cfk->bfk',x.reshape(len(x),-1),st[f'w{j}'])
        z=np.einsum('tsf,tfk->tsk',f,v[np.arange(len(f))//BLOCK])
        z[:,:,0]=np.roll(z[:,:,0],13,axis=1)
        out.append(.30*z)
    return np.concatenate(out).astype(np.float32)


def run():
    geo=geometry()
    hist=[]; cl=[]; ph=[]; obs=[]; ds=[]
    for d in range(4,16):
        raw,refs,y=parent.history.training_day(d)
        refs=np.array(refs,copy=True)
        base=[parent.exposure.observations(p,y) for p in refs]
        if d>=8:
            if d%2==0:
                state=fit(hist); common=clock.params(cl); pw=portfolios.fit(ph); w=parent.exposure.fit(obs)[6]
            refs[:,:,:,2]+=parent.apply_market(*parent.historical_heads(d),w)
            group=[]
            strict=np.load(parent.ART/'cutoff_reference_panel_v1'/f'panel_{d}.npy')
            for j,p in enumerate([*refs,strict]):
                z=signals(raw,p,state,geo)
                if j==0 and d==8:
                    for n in (37,154,2476):
                        np.testing.assert_allclose(z[:n],signals(raw[:n],p[:n],state,geo),atol=4e-7,rtol=0)
                q=p.copy()
                q[:,:,2]+=clock.signals(raw,p,common)[:,:,clock.CHOSEN-1]+portfolios.corrections(raw,p,pw)[:,:,2]
                group.append(parent.base.records(q,y,z))
            ds.append(group)
        obs.append(tuple(np.mean(v,0) for v in zip(*base)))
        for dst,data in ((hist,[record(raw,p,y,geo,d) for p in refs]),(cl,[clock.record(raw,p,y) for p in refs]),
                         (ph,[portfolios.record(raw,p,y,d) for p in refs])):
            dst.append([tuple(np.mean(v,0) for v in zip(*s)) for s in zip(*data)])
        print('completed training-period day',d+1,flush=True)
    def scores(group):return np.array([parent.base.score(m,np.ones(len(ARMS)-1)) for m in np.sum(group,0)])
    h=np.stack([scores(ds[:4]),scores(ds[4:])])
    daily=np.stack([scores([d]) for d in ds])
    pick=max(range(2,len(ARMS)),key=lambda i:h[0,:,:,i].min())
    ok=h[:,:,:,pick].min()>0 and (daily[:,:,0,pick]>0).sum(0).min()>=5 and (h[:,:,0,pick]>h[:,:,0,1]).all()
    OUT.mkdir(exist_ok=True)
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,windows=[[9,12],[13,16]],first_window_choice=pick,
        selected=pick if ok else 0,halves=h.tolist(),daily=daily.tolist()),indent=2))
    if ok:np.savez(OUT/'model.npz',pick=pick,k=geo[0],coef=geo[1],row=geo[2],**fit(hist))
    print('selected',pick if ok else 0,'strict Y3',h[:,-1,0].round(6).tolist(),flush=True)


def correction():
    before=portfolios.correction()
    with np.load(OUT/'model.npz') as f:st=dict(f)
    geo=float(st['k']),st['coef'],st['row']
    return lambda raw,p:before(raw,p)+signals(raw,p,st,geo)[:,:,int(st['pick'])-1]


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('train','screen','verify'),nargs='?',default='train')
    mode=parser.parse_args().mode
    if mode=='train':run()
    else:evaluate(parent,OUT,correction,mode=='screen')
