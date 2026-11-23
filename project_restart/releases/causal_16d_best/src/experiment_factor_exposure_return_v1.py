"""Causal, exposure-aware correction of the market and factor components."""
import json
import numpy as np
import torch
import experiment_multiscale_return_net_v1 as net
import cache_opening_walkforward as opening
from experiment_causal_return_basket_v1 import grid,moving_sum
prior=net.prior
OUT=prior.ART/'factor_exposure_return_v1'


def v1_base(day):
    if day<12:return np.asarray(opening.load(day))
    if day>=16:return np.load(net.OUT/f'pred_{day}.npy')
    fp=OUT/f'v1_{day}.npy'
    if fp.exists():return np.load(fp)
    end=12 if day<14 else 14
    cfg=json.loads((net.OUT/'results.json').read_text())['16']
    feat=np.load(net.OUT/f'features_{day}.npy',mmap_mode='r')
    side=np.zeros((len(feat),50,3),np.float32)
    with torch.inference_mode():
        for e in set(cfg['epochs']):
            model=net.ReturnNet(feat.shape[-1]).eval()
            model.load_state_dict(torch.load(net.OUT/f'net_{end}_{e}.pth',map_location='cpu',weights_only=True))
            js=[i for i,v in enumerate(cfg['epochs']) if v==e]
            for i in range(0,len(feat),128):
                inp=torch.from_numpy(feat[i:i+128].astype(np.float32))
                side[i:i+128,:,js]=net.decode(model(inp)).numpy()[:,:,js]
    base=net.blend(opening.load(day),side,np.array(cfg['weights'])).astype(np.float32)
    np.save(fp,base)
    return base


def geometry(end):
    cov=[]
    for d in range(end):
        r=grid(d)[:,:,0].astype(float)
        obs=moving_sum(r,40)[39::40]
        cov.append(np.cov(obs.T))
    cov=np.mean(cov,axis=0)
    f=cov.mean(1)/cov.mean()
    scale=np.sqrt(cov.diagonal())
    # Separate broad market beta, top relative return modes and residual structure.
    cm=cov-np.outer(f,f)*cov.mean()
    std=np.sqrt(cm.diagonal())
    vals,vec=np.linalg.eigh(cm/np.outer(std,std))
    p=(vec[:,-4:]@vec[:,-4:].T)*np.outer(std,1/std)
    np.fill_diagonal(p,0)
    return f,scale/scale.mean(),p


def windows(x,k):
    t=np.arange(1,len(x)+1)
    return moving_sum(x,k)/np.minimum(t,k).reshape((-1,)+(1,)*(x.ndim-1))


def features(raw,base,geo):
    f,sc,proj=geo
    p=base.astype(float);r=np.asarray(raw[:,:,0],float)
    rv=np.sqrt(.01+windows(r*r,160))
    rms=(.35+.65*rv).clip(.5,4.)
    market=r.mean(1,keepdims=True)
    # Dynamic beta from past completed 10-tick returns, shrunk toward the train estimate.
    R=moving_sum(r,10)/np.sqrt(10)
    M=R.mean(1,keepdims=True)
    dyn=(moving_sum(R*M,480)+24*f)/(moving_sum(M*M,480)+24)
    cols=[]
    for v in (p,windows(p,12),windows(p,60)):
        avg=v.mean(1,keepdims=True)
        rel=v-avg
        smarket=(v/sc[None,:,None]).mean(1,keepdims=True)
        lead=np.einsum('ij,tjk->tik',proj,rel,optimize=True)
        for val in ((f-1)[None,:,None]*avg,(dyn-1)[:,:,None]*avg,(sc-1)[None,:,None]*avg,
                    smarket*sc[None,:,None]-avg,lead,
                    (rms/rms.mean(1,keepdims=True)-1)[:,:,None]*avg):
            cols.append(val-val.mean(1,keepdims=True))
    for k in (2,8,24,60,160):
        own=moving_sum(r,k)/np.sqrt(k)
        m=own.mean(1,keepdims=True);rel=own-m*f
        fm=(f[None,:]-1)*m
        field=np.stack((fm,(dyn-f)*m,rel@proj.T),axis=-1)
        cols.append(field-field.mean(1,keepdims=True))
    return np.concatenate(cols,axis=-1).astype(np.float32)


def fit(days,geo):
    mats=[]
    for d in days:
        p=v1_base(d);y=prior.labels(d).reshape(p.shape)
        e=y-p;e-=e.mean(1,keepdims=True)
        x=features(grid(d),p,geo)[::3].reshape(-1,69).astype(float)
        t=e[::3].reshape(-1,3)
        g=x.T@x/len(x);b=x.T@t/len(x)
        mats.append((g,b))
    gram,rhs=(np.mean(v,axis=0) for v in zip(*mats))
    reg=.7*np.diag(np.maximum(np.diag(gram),1e-6))
    general=np.linalg.solve(gram+reg,rhs)
    stable=np.stack([np.linalg.solve(g+reg,b) for g,b in mats])
    return .5*general+.5*np.median(stable,axis=0)


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    early=[];report={}
    for end,stop in [(12,14),(14,16),(16,20)]:
        geo=geometry(end);w=fit(range(8,end),geo)
        rows=[]
        for d in range(end,stop):
            p=v1_base(d);y=prior.labels(d).reshape(p.shape)
            x=features(grid(d),p,geo)
            if d==end:
                np.testing.assert_allclose(x[:79],features(grid(d)[:79],p[:79],geo),rtol=2e-5,atol=2e-5)
            rows.append((p,y,(x@w).astype(np.float32)))
        if end<16:
            early.extend(rows)
            p,y,s=map(np.concatenate,zip(*early))
            best=np.full(3,-np.inf);weights=np.zeros(3)
            for gain in (0,.25,.5,1.):
                corr=prior.corr((p+gain*s).reshape(-1,3),y.reshape(-1,3))
                ix=corr>best;weights[ix]=gain;best=np.maximum(best,corr)
        else:
            np.savez(OUT/'model.npz',beta=geo[0],scale=geo[1],proj=geo[2],coef=w,weights=weights)
        entries=[]
        for d,(p,y,s) in zip(range(end,stop),rows):
            q=p+weights*s
            np.testing.assert_allclose(q.mean(1),p.mean(1),atol=2e-6)
            if end==16:np.save(OUT/f'pred_{d}.npy',q.astype(np.float32))
            entries.append((p.reshape(-1,3),q.reshape(-1,3),y.reshape(-1,3)))
        report[stop]={'weights':weights.tolist(),'result':prior.measure(entries)}
        (OUT/'results.json').write_text(json.dumps(report,indent=2))
        print(stop,report[stop],flush=True)


if __name__=='__main__':main()
