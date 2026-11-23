"""Adjust a single expert's contribution continuously using causal opening phase and realized volatility."""
import json
import numpy as np
import experiment_endpoint_innovation_v1 as inn
import experiment_phase_future_book_v1 as phase
prior=inn.prior
OUT=prior.ART/'phase_conditional_blend_v1'

def states(raw,base,side):
    n=len(raw);tick=np.arange(n);r=raw[:,:,0].astype(float)-.00136
    v=np.sqrt((10+inn.moving_sum(r*r,96))/(10+np.minimum(tick+1,96))[:,None])
    lv=np.log(v+.15).clip(-1.5,1.5)
    opening=np.exp(-tick/330)[:,None,None]
    d=side.astype(float)-base;dm=d.mean(1,keepdims=True);dr=d-dm
    market=[dm*opening,dm*lv.mean(1)[:,None,None]]
    relative=[dr*opening,dr*lv[:,:,None]]
    return [np.broadcast_to(x,base.shape) for x in market]+[x-x.mean(1,keepdims=True) for x in relative]

def moments(p,y,z):
    y=y.reshape(-1,3).astype(float);p=p.reshape(-1,3).astype(float)
    x=np.stack(z,axis=-1).reshape(-1,3,len(z))
    x-=x.mean(0);y-=y.mean(0);p-=p.mean(0)
    var=(p*p).mean(0);cov=(p*y).mean(0);slope=(cov/var).clip(.4,2.5)
    # Remove the scale direction to fit the local change in Pearson correlation.
    projection=np.einsum('nki,nk->ki',x,p)/len(y)/var[:,None]
    x-=p[:,:,None]*projection[None]
    e=y/slope-p
    return np.einsum('nki,nkj->kij',x,x)/len(x),np.einsum('nki,nk->ki',x,e)/len(x)

def select(hist):
    grams,bs=[np.mean(v,axis=0) for v in zip(*hist)]
    coefs=[]
    for k,(g,b) in enumerate(zip(grams,bs)):
        reg=.80*np.diag(np.maximum(2e-5,g.diagonal()))
        joint=np.linalg.solve(g+reg,b)
        daily=np.stack([np.linalg.solve(a[k]+reg,t[k]) for a,t in hist])
        coefs.append(.6*joint+.4*np.median(daily,axis=0))
    return (.65*np.array(coefs)).clip(-.22,.28)

def run():
    OUT.mkdir(exist_ok=True);hist=[];rows=[];report={}
    for d in range(12,20):
        if d in (14,16):
            if d==16 and report[16]['gain']<=0:break
            coef=select(hist);rows=[]
            if d==16:np.save(OUT/'coef.npy',coef)
        p=np.load(phase.OUT/f'pred_{d}.npy');base=np.load(phase.ref.OUT/f'pred_{d}.npy')
        side=np.load(phase.OUT/f'side_{d}.npy');raw=inn.grid(d)
        z=states(raw,base,side);y=prior.labels(d).reshape(p.shape)
        if d>=14:
            q=p+sum(zj*c for zj,c in zip(z,coef.T));np.save(OUT/f'pred_{d}.npy',q.astype(np.float32))
            rows.append(tuple(a.reshape(-1,3) for a in (p,q,y)))
        if d<16:hist.append(moments(p,y,z))
        if d in (15,19):
            report[d+1]=prior.measure(rows);(OUT/'results.json').write_text(json.dumps(report,indent=2));print(d+1,coef,report[d+1],flush=True)
if __name__=='__main__':run()
