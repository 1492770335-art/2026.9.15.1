"""Reduced-rank cross-asset price impact with train-only return factor portfolios."""
import json
import numpy as np
import experiment_factor_exposure_return_v1 as ex
import experiment_peer_return_distillation_v1 as fits
import experiment_multiscale_return_net_v1 as v1
from experiment_causal_return_basket_v1 import moving_sum
prior=v1.prior
OUT=prior.ART/'pooled_price_impact_v1'


def states(raw):
    x=raw.astype(np.float32)
    a=np.arcsinh(x)
    def avg(cols):return a[:,:,cols].mean(2)
    pressure=np.stack([a[:,:,0],avg([7,8,9]),avg([10,11,12]),avg([13,14,15]),
                       avg([20,21,22,23]),avg([24,25,26,27]),avg([28,29,30,31]),
                       a[:,:,41]-a[:,:,42],a[:,:,38],a[:,:,50]-a[:,:,51],
                       a[:,:,52]-a[:,:,53]+a[:,:,54]-a[:,:,55],a[:,:,56]-a[:,:,57],
                       avg([58,59,60,61,62,63,64,65]),avg([66,67,68,69,70,71,72,73]),
                       avg([74,75,76]),x[:,:,84]-1.,x[:,:,87]-1.,x[:,:,94]-1.,
                       a[:,:,16],a[:,:,17],a[:,:,5]],axis=2)
    n=len(x);t=np.arange(n)+1
    ss=np.concatenate((np.zeros_like(pressure[:1],dtype=float),pressure.cumsum(0,dtype=float)))
    parts=[pressure]
    for low,high in ((1,5),(5,20),(20,60),(60,180)):
        u=np.maximum(0,t-low);v=np.maximum(0,t-high)
        parts.append(((ss[u]-ss[v])/np.maximum(1,u-v)[:,None,None]).astype(np.float32))
    return np.concatenate(parts,axis=-1)


def exposure(end):
    mat=[]
    for d in range(end):
        r=moving_sum(v1.grid(d)[:,:,0].astype(float),30)[29::30]
        mat.append(np.cov(r.T))
    cov=np.mean(mat,axis=0);scale=np.sqrt(cov.diagonal())
    corr=cov/np.outer(scale,scale)
    _,q=np.linalg.eigh(corr)
    return q[:,-6:].T/np.sqrt(50),scale/scale.mean()


def build(z,proj):
    # Shared microstructure slopes for every asset; common factor time series
    # retain signed cross-asset information separately for each pressure band.
    return np.einsum('ks,tsf->tkf',proj,z,optimize=True).reshape(len(z),-1)


def solve(g,rhs,alpha):
    return np.linalg.solve(g+alpha*np.diag(np.maximum(np.diag(g),1e-6)),rhs)


def train(end,proj,vol):
    shared=[];factor=[]
    for d in range(end):
        z=states(v1.grid(d))[d%4::4].astype(float)
        y=prior.labels(d).reshape(-1,50,3)[d%4::4].astype(float)
        zz=z.reshape(-1,z.shape[-1]); t=(y/vol[None,:,None]).reshape(-1,3)
        x=build(z,proj)
        # Fit a global response on residuals conditional on shared local model.
        shared.append((zz.T@zz/len(zz),zz.T@t/len(zz)))
        factor.append((x.T@x/len(x),x.T@t.reshape(len(x),-1)/len(x),
                       (x.T@zz.reshape(len(x),-1)/len(x)).reshape(x.shape[1],50,-1)))
    gm,bm=(np.mean(v,0) for v in zip(*shared))
    local=solve(gm,bm,.075)
    g,b,c=(np.mean(v,0) for v in zip(*factor))
    target=b-(c@local).reshape(b.shape)
    cross=solve(g,target,.6)
    return local,cross


def main():
    OUT.mkdir(exist_ok=True);early=[];report={}
    for end,stop in [(12,14),(14,16),(16,20)]:
        proj,vol=exposure(end);local,cross=train(end,proj,vol)
        rows=[]
        for d in range(end,stop):
            z=states(v1.grid(d))
            if d==end:np.testing.assert_allclose(z[:97],states(v1.grid(d)[:97]),atol=1e-6)
            n=((z@local)+(build(z,proj)@cross).reshape(len(z),50,3))*vol[None,:,None]
            p=ex.v1_base(d);y=prior.labels(d).reshape(p.shape)
            rows.append((p,y,n.astype(np.float32)))
        if end<16:
            early.extend(rows);_,w=fits.coefficients(early)
        else:np.savez(OUT/'model.npz',proj=proj,vol=vol,local=local,cross=cross,blend=w)
        entries=[]
        for d,(p,y,n) in zip(range(end,stop),rows):
            q=v1.blend(p,n,w)
            np.save(OUT/f'side_{d}.npy',n)
            entries.append((p.reshape(-1,3),q.reshape(-1,3),y.reshape(-1,3)))
        report[stop]={'weights':w.tolist(),'result':prior.measure(entries)}
        (OUT/'results.json').write_text(json.dumps(report,indent=2))
        print('score',stop,report[stop],flush=True)

if __name__=='__main__':main()
