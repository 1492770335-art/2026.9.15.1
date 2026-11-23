"""Causal intraday calibration using market forecasts whose return windows have already elapsed."""
import json
import numpy as np
import experiment_endpoint_innovation_v1 as inn
import experiment_relative_residual_tree_v1 as tree
from experiment_multiscale_return_net_v1 import SCALE
prior=inn.prior
OUT=prior.ART/'matured_market_innovations_v1'

def design(raw,p):
    t=np.arange(len(p));r=raw[:,:,0].astype(float)-.00136
    market=r.mean(1);m=p.mean(1)
    z=[np.ones(len(t)),m[:,0]*5,m[:,1]*5,m[:,2]*5]
    for h in (3,16,60):
        z.append(inn.moving_sum(market,h)/np.sqrt(h)*3)
    for j in [11,16,26,38,41,42,64,94]:
        a=np.arcsinh(raw[:,:,j].astype(float));v=a.mean(1)
        z.append((v-inn.moving_sum(v,200)/np.minimum(t+1,200))*4)
    return np.column_stack(z).clip(-4,4),market,m

def adapt(raw,p):
    x,r,m=design(raw,p);t=np.arange(len(p));out=[]
    for width,shrink in [(256,900),(1200,1500),(4096,2000)]:
        side=np.zeros_like(m)
        for j,h in enumerate(inn.H):
            y=inn.moving_sum(r,h)*SCALE[j]
            origin=np.maximum(0,t-h)
            hist=x[origin].copy();hist[:h]=0
            err=(y-m[origin,j]).clip(-1.8,1.8)
            # At tick t these observations came from predictions issued at t-h or earlier.
            gram=inn.moving_sum(hist[:,:,None]*hist[:,None,:],width)
            rhs=inn.moving_sum(hist*err[:,None],width)
            gram+=np.eye(x.shape[1])[None]*shrink*.55
            beta=np.linalg.solve(gram,rhs[:,:,None])[:,:,0]
            side[:,j]=(x*beta).sum(1).clip(-.45,.45)
        out.append(side)
    return np.stack(out,axis=2)

def main():
    OUT.mkdir(exist_ok=True);hist=[];report={};rows=[]
    for d in range(8,20):
        if d in (12,14,16):
            gram=np.mean([a for a,b in hist],0);rhs=np.mean([b for a,b in hist],0)
            w=np.stack([np.linalg.solve(g+.65*np.diag(np.maximum(1e-6,g.diagonal())),b) for g,b in zip(gram,rhs)])
            w=np.clip(w,-1.,1.)
            if d==16:np.save(OUT/'weights.npy',w)
            rows=[]
        p=inn.forecast(d) if d<12 else np.load(tree.OUT/f'pred_{d}.npy').astype(float)
        raw=inn.grid(d);y=prior.labels(d).reshape(p.shape);h=adapt(raw,p)
        if d in (12,16):np.testing.assert_allclose(h[:273],adapt(raw[:273],p[:273]),atol=2e-8)
        if d>=12:
            q=p+np.einsum('tkj,kj->tk',h,w)[:,None,:];np.save(OUT/f'pred_{d}.npy',q.astype(np.float32))
            rows.append(tuple(a.reshape(-1,3) for a in (p,q,y)))
        if d<16:
            e=(y-p).mean(1);hist.append((np.einsum('tki,tkj->kij',h,h)/len(h),np.einsum('tki,tk->ki',h,e)/len(h)))
        if d in (13,15,19):
            report[d+1]={'w':w.tolist(),'result':prior.measure(rows)}
            (OUT/'results.json').write_text(json.dumps(report,indent=2));print(d+1,report[d+1],flush=True)
if __name__=='__main__':main()
