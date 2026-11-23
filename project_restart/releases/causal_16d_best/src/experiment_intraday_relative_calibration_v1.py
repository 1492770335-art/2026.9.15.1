"""Update stock-relative forecast corrections from matured within-day outcomes with hierarchical shrinkage."""
import json
import numpy as np
import experiment_endpoint_innovation_v1 as inn
from experiment_matured_market_innovations_v1 import prior
OUT=prior.ART/'intraday_relative_calibration_v1'
REFERENCE=prior.ART/'curve_quantile_market_v1'
BLOCK=12
def forecast(d):
    if d<12:return inn.forecast(d)
    return np.load(REFERENCE/f'pred_{d}.npy').astype(float)

def context(raw,p):
    t=np.arange(len(raw))
    r=raw[:,:,0].astype(float)-.00136
    vol=np.sqrt((8+inn.moving_sum(r*r,96))/(np.minimum(t+1,96)+8)[:,None]).clip(.45,3.5)
    rel=p-p.mean(1,keepdims=True)
    z=[np.ones_like(r),*(rel[:,:,j]/vol*2.5 for j in range(3))]
    z.extend([np.broadcast_to(p[:,:,j].mean(1,keepdims=True),r.shape)*3 for j in (1,2)])
    z.extend([np.arcsinh(raw[:,:,j].astype(float)) for j in (11,26,38,41,42,60,94)])
    for h in (5,40):
        move=inn.moving_sum(r,h)/np.sqrt(h)
        z.append((move-move.mean(1,keepdims=True))/vol)
    return np.stack(z,axis=2).clip(-3.5,3.5),vol,r

def rolling(a,n):
    # A block is usable only after it has finished.
    s=np.concatenate((np.zeros_like(a[:1]),a.cumsum(axis=0)))
    i=np.arange(len(a))
    return s[i]-s[np.maximum(0,i-n)]

def adaptive(raw,p):
    z,vol,r=context(raw,p);n=len(p);t=np.arange(n);nb=(n+BLOCK-1)//BLOCK
    rel=p-p.mean(1,keepdims=True)
    out=np.zeros((n,50,3,4),np.float32)
    dim=z.shape[-1]
    for j,h in enumerate(inn.H):
        origin=np.maximum(0,t-h)
        y=inn.moving_sum(r,h)*inn.SCALE[j]
        residual=(y-y.mean(1,keepdims=True)-rel[origin,:,j])/vol[origin]
        residual=np.clip(residual,-2,2)
        x=z[origin].copy();x[:h]=0
        padded=np.zeros((nb*BLOCK,50,dim));padded[:n]=x
        er=np.zeros((nb*BLOCK,50));er[:n]=residual
        x=padded.reshape(nb,BLOCK,50,dim);er=er.reshape(nb,BLOCK,50)
        grams=np.einsum('btsi,btsj->bsij',x,x,optimize=True);rhs=np.einsum('btsi,bts->bsi',x,er)
        for k,(memory,asset_strength) in enumerate([(40,420),(165,1100)]):
            g=rolling(grams,memory);b=rolling(rhs,memory)
            gs=g.sum(1);bs=b.sum(1)
            # Shrink toward the fixed forecast; effective samples are fewer than overlapping ticks.
            shared=np.linalg.solve(gs+np.eye(dim)[None]*50*asset_strength,bs[:,:,None])[:,:,0]
            personal=np.linalg.solve(g+np.eye(dim)[None,None]*asset_strength*2.5,
                        (b-(g@shared[:,None,:,None])[:,:,:,0])[:,:,:,None])[:,:,:,0]
            for c,coef in enumerate([shared[:,None,:],personal]):
                correction=np.sum(z*coef[t//BLOCK],axis=2).clip(-1.5,1.5)*vol
                out[:,:,j,2*k+c]=correction-correction.mean(1,keepdims=True)
    return out

def main():
    OUT.mkdir(exist_ok=True);hist=[];report={};early=[];outer=[]
    for d in range(8,20):
        if d in (12,14,16):
            grams,bs=[np.mean(v,axis=0) for v in zip(*hist)]
            coefs=np.stack([np.linalg.solve(g+.9*np.diag(np.maximum(g.diagonal(),1e-6)),b) for g,b in zip(grams,bs)]).clip(-1.,1.)
            if d==16:
                if report[16]['gain']<=0:break
                np.save(OUT/'coef.npy',coefs)
        p=forecast(d);y=prior.labels(d).reshape(p.shape);raw=inn.grid(d)
        x=adaptive(raw,p)
        if d in (12,16):np.testing.assert_allclose(x[:347],adaptive(raw[:347],p[:347]),atol=2e-6)
        if d>=12:
            q=p+np.einsum('tskj,kj->tsk',x,coefs);q=q.astype(np.float32)
            np.save(OUT/f'pred_{d}.npy',q)
            outer.append(tuple(v.reshape(-1,3) for v in (p,q,y)))
        if d<16:
            err=y-p;err-=err.mean(1,keepdims=True)
            hist.append((np.einsum('tski,tskj->kij',x,x,dtype=float)/len(p)/50,
                         np.einsum('tski,tsk->ki',x,err,dtype=float)/len(p)/50))
        if d in (13,15,19):
            report[d+1]=prior.measure(outer)
            (OUT/'results.json').write_text(json.dumps(report,indent=2))
            print(d+1,report[d+1],flush=True);outer=[]
if __name__=='__main__':main()
