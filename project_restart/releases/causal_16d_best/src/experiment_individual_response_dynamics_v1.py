"""Track stock-specific changes in the long-return response, borrowing strength across dates and stocks."""
import json
import numpy as np
import experiment_supervised_gradient_kernel_v1 as kernel
import experiment_rolling_drift_strength_v1 as prev
import experiment_delayed_tail_band_skill_v1 as ref
import experiment_pairwise_book_tree_v1 as pair
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,corr,measure
from experiment_multiscale_return_net_v1 import SCALE

OUT=ART/'individual_response_dynamics_v1'
RATIO=float(SCALE[2]/SCALE[1])


def prior():
    geo=kernel.geometry();w=kernel.fit([kernel.statistics(d,geo) for d in range(8)])[1]
    return geo[1],w


def frame(s,v,t,g,w):
    _,r=kernel.panel(s,v,t);n=r.shape[-1];x=((r-g['center'])/g['scale']).clip(-4,4)
    proj=x@g['omega'];ss=np.sin(proj);cc=np.cos(proj);k=ss.shape[-1]
    linear=w[1:n+1];ws=w[n+1:n+1+k];wc=w[n+1+k:]
    parts=[x[:,:,:31]@linear[:31],x[:,:,31:70]@linear[31:70],x[:,:,70:]@linear[70:]]
    for start in range(4):
        sl=slice(start,None,4)
        parts.append(ss[:,:,sl]@ws[sl]+cc[:,:,sl]@wc[sl])
    parts.extend((np.take(x,(0,3,5,7,11,12,23,28,29,30,40,53,66),axis=-1),np.log(v[t])[:,:,None]))
    return np.concatenate(parts,axis=2).astype('float32')


def moments(d,g,w):
    raw=pair.rawday(d);s,v=pair.state(raw);t=np.arange(d%7,len(s)-61,7)
    y=labels(d).reshape(-1,50,3)[t];y-=y.mean(1,keepdims=True)
    y=np.concatenate((y,y[:,:,2:3]-RATIO*y[:,:,1:2]),axis=2)/(.45+.55*v[t,:,None])
    z=frame(s,v,t,g,w);y=y.clip(-4.5,4.5)
    sets=[]
    for half in (0,1):
        ix=(t//250%2==half)&(t%250<180)
        x=np.ascontiguousarray(z[ix].transpose(1,0,2));ys=np.ascontiguousarray(y[ix].transpose(1,0,2))
        xt=x.transpose(0,2,1);sets.append((xt@x/ix.sum(),xt@ys/ix.sum()))
    return sets


def fit(history):
    gs,bs=(np.array(z,dtype=float) for z in zip(*[h for d in history for h in d]))
    n=len(history);gs=gs.reshape(n,2,*gs.shape[1:]);bs=bs.reshape(n,2,*bs.shape[1:])
    gram=gs.mean((0,1,2));rhs=bs.mean((0,1,2))
    ridge=.60*np.diag(gram.diagonal().clip(.00008,None))
    fixed=np.linalg.solve(gram+ridge,rhs);wh=np.linalg.inv(np.linalg.cholesky(gram+ridge))
    grad=np.einsum('ij,dhsjk->dhsik',wh,bs-gs@fixed,optimize=True)
    grad-=grad.mean(2,keepdims=True)
    responses=[];drifts=[]
    for k in range(4):
        a,b=grad[:,0,:,:,k],grad[:,1,:,:,k]
        aa=a.reshape(-1,a.shape[-1]);bb=b.reshape(aa.shape)
        cross=(aa.T@bb+bb.T@aa)/(2*len(aa))
        val,axes=np.linalg.eigh(cross);axes=axes[:,-6:]
        za=a@axes;zb=b@axes;state=.5*(za+zb)
        avg=state.mean(0)
        first=state[::2].mean(0);second=state[1::2].mean(0)
        between=(first*second).mean(0).clip(0,None)
        total=(avg**2).mean(0).clip(1e-7,None)
        stable=(between/(total+.0001)).clip(0,.90)
        # Estimate one-day persistence on pooled stock trajectories after removing their long-term level.
        xa=za-stable*avg;xb=zb-stable*avg
        sig=(xa*xb).mean((0,1)).clip(.00002,None)
        noise=.25*((xa-xb)**2).mean((0,1))
        lag=.5*(xa[:-1]*xb[1:]+xb[:-1]*xa[1:]).mean((0,1))
        rho=(lag/(sig+noise+.0001)).clip(-.48,.68)*n/(n+3)
        last=state-stable*avg
        drift=rho*(.72*last[-1]+.28*rho*last[-2])*sig/(sig+noise+.0001)
        basis=wh.T@axes
        responses.append((stable*avg)@basis.T)
        drifts.append(drift@basis.T)
    return tuple(np.stack(z,axis=2).astype('float32') for z in (responses,drifts))


def signals(raw,p,g,w,weights):
    s,v=pair.state(raw);out=[np.empty((*p.shape[:2],4),np.float32) for _ in weights]
    for lo in range(0,len(raw),94):
        t=np.arange(lo,min(lo+94,len(raw)));x=frame(s,v,t,g,w)
        for o,params in zip(out,weights):
            resp=np.einsum('tsd,sdh->tsh',x,params,optimize=True)*(.45+.55*v[t,:,None])
            o[t]=pair.innovation(p[t],resp-resp.mean(1,keepdims=True))
    return out


def main():
    OUT.mkdir(exist_ok=True);g,w=prior();history=[];hist=[[],[]];daily=[];report={};ok=np.ones(2,bool)
    for end,stop in ((8,10),(10,12),(12,14),(14,16),(16,20)):
        history.extend(moments(d,g,w) for d in range(len(history),end));params=fit(history);rows=[]
        if end>=10:
            if end==16:
                gains=np.stack(daily);ok=((gains>0).sum(0)>=4)&(gains.mean(0)>0)&(gains[-2:].mean(0)>0)
                report['eligible']=ok.tolist()
                if not ok.any():break
            coef=np.stack([pair.weights(z) for z in hist])*ok[:,None]
            if end==16:np.savez(OUT/'model.npz',**g,w=w,stable=params[0],drift=params[1],coef=coef)
        for d in range(end,stop):
            p=np.load(prev.OUT/f'pred_{d}.npy') if d>=16 else ref.base(d);raw=pair.rawday(d);xx=signals(raw,p,g,w,params)
            if d==end:
                small=signals(raw[:193],p[:193],g,w,params)
                for a,b in zip(xx,small):np.testing.assert_allclose(a[:193],b,atol=3e-6)
            if end>=10:
                terms=[a@b for a,b in zip(xx,coef)]
                q=p.copy();q[:,:,2]+=sum(terms)
                if d>=16:np.save(OUT/f'pred_{d}.npy',q)
            y=labels(d).reshape(p.shape)
            if d<16:
                a=p[:,:,2].astype(float);a-=a.mean();b=y[:,:,2].astype(float);b-=b.mean()
                slope=((a*b).sum()/(a*a).sum()).clip(.4,2.5);error=b/slope-a;error-=error.mean(1,keepdims=True)
                for h,x in zip(hist,xx):
                    flat=x.reshape(-1,4).astype(float);h.append((flat.T@flat/len(flat),flat.T@error.ravel()/len(flat)))
            if end>=10:
                row=tuple(v.reshape(-1,3) for v in (p,q,y));rows.append(row)
                if d<16:
                    target=y[:,:,2].ravel();old=np.corrcoef(p[:,:,2].ravel(),target)[0,1]
                    daily.append([np.corrcoef((p[:,:,2]+a).ravel(),target)[0,1]-old for a in terms])
        if end>=10:report[str(stop)]=measure(rows);print(stop,report[str(stop)],flush=True)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
