"""Expanded low-rank order-flow regression with complementary stock-specific source alphas."""
import json
import numpy as np
import experiment_pooled_price_impact_v1 as first
import experiment_factor_exposure_return_v1 as ex
import experiment_multiscale_return_net_v1 as v1
import experiment_peer_return_distillation_v1 as fits
prior=v1.prior
OUT=prior.ART/'source_alpha_impact_v1'


def states(raw):
    z=first.states(raw)
    a=np.arcsinh(raw.astype(np.float32))
    flow=np.stack((a[:,:,41]-a[:,:,42], a[:,:,26],a[:,:,11],a[:,:,56]-a[:,:,57],
                   a[:,:,64],a[:,:,75],a[:,:,37],a[:,:,1]-a[:1,:,1],a[:,:,3]+a[:,:,4],
                   a[:,:,16],a[:,:,39]-a[:,:,40],a[:,:,7]),axis=2)
    n=len(raw); t=np.arange(n)+1
    cum=np.concatenate((np.zeros_like(flow[:1],dtype=float),flow.cumsum(0,dtype=float)))
    price=raw[:,:,0].astype(float)
    sq=np.vstack((np.zeros(50),(price*price).cumsum(0)))
    vol=np.sqrt((sq[t]-sq[np.maximum(0,t-96)]+4)/(np.minimum(t,96)+4)[:,None]).clip(.4,3.5)
    recent=np.empty_like(flow)
    l=np.maximum(0,t-8); recent[:]=(cum[t]-cum[l])/(t-l)[:,None,None]
    b=np.maximum(0,t-90);old=((cum[t]-cum[b])/(t-b)[:,None,None]).astype(np.float32)
    vol=vol.astype(np.float32)[:,:,None]
    return np.concatenate((z,flow,flow/vol,recent*vol,(flow-old)/vol,np.tanh(recent),old*vol),axis=2)


def normal(g,alpha):
    return alpha*np.diag(np.maximum(np.diag(g),1e-7))


def fit(end):
    proj,vol=first.exposure(end);local_mom=[];correlations=[]
    for d in range(end):
        z=states(v1.grid(d))[d%4::4].astype(float)
        y=prior.labels(d).reshape(-1,50,3)[d%4::4]
        t=y/vol[None,:,None];flat=z.reshape(-1,z.shape[-1])
        local_mom.append((flat.T@flat/len(flat),flat.T@t.reshape(-1,3)/len(flat)))
        # Per-stock transfer to the following market return, de-mean each day.
        centered=z-z.mean(0,keepdims=True)
        ym=y.mean(1);ym=(ym-ym.mean(0))/ym.std(0)
        cross=np.einsum('tsf,tk->sfk',centered,ym,optimize=True)/len(ym)
        correlations.append(cross/np.maximum(centered.std(axis=0)[:,:,None],.05))
    g,b=(np.mean(v,0) for v in zip(*local_mom))
    local=np.linalg.solve(g+normal(g,.07),b)
    a=np.mean(correlations,axis=0).reshape(50,-1)
    q=proj.T*np.sqrt(50)
    a-=q@(q.T@a)
    u,_,_=np.linalg.svd(a,full_matrices=False)
    source=u[:,:3].T/np.sqrt(50)
    mom=[]
    for d in range(end):
        z=states(v1.grid(d))[d%4::4].astype(float)
        y=prior.labels(d).reshape(-1,50,3)[d%4::4]/vol[None,:,None]
        loc=z@local
        x=design(z,loc,proj,source)
        yy=(y-loc).reshape(len(y),-1)
        mom.append((x.T@x/len(x),x.T@yy/len(x)))
    g,b=(np.mean(v,0) for v in zip(*mom))
    # Heavier shrinkage for learned source directions and all-asset response channels.
    n=proj.shape[0]*len(local)
    penalty=np.r_[np.full(n,.5),np.full(g.shape[0]-n,1.6)]
    coef=np.linalg.solve(g+normal(g,penalty),b)
    return proj,vol,local,source,coef


def design(z,loc,proj,source):
    # Only learned price pressure, instead of every raw field, bypasses the
    # six-factor restriction. This caps the number of cross-asset parameters.
    return np.column_stack((first.build(z,proj),first.build(z[:,:,:105],source),loc.reshape(len(z),-1)))


def main():
    OUT.mkdir(exist_ok=True)
    report={};early=[]
    previous=json.loads((first.OUT/'results.json').read_text())['16']
    for end,stop in [(12,14),(14,16),(16,20)]:
        proj,vol,local,source,coef=fit(end)
        rows=[]
        for d in range(end,stop):
            x=v1.grid(d);z=states(x)
            if d==end:np.testing.assert_allclose(z[:103],states(x[:103]),atol=1e-6)
            loc=z@local
            n=(loc+(design(z,loc,proj,source)@coef).reshape(len(z),50,3))*vol[None,:,None]
            p=ex.v1_base(d)
            older=np.load(first.OUT/f'side_{d}.npy')
            base=v1.blend(p,older,np.array(previous['weights'])).astype(np.float32)
            rows.append((base,prior.labels(d).reshape(p.shape),n.astype(np.float32)))
        if end<16:early.extend(rows);_,w=fits.coefficients(early)
        else:np.savez(OUT/'model.npz',proj=proj,vol=vol,local=local,source=source,coef=coef,blend=w)
        entries=[]
        for d,(p,y,n) in zip(range(end,stop),rows):
            q=v1.blend(p,n,w)
            np.save(OUT/f'side_{d}.npy',n);np.save(OUT/f'pred_{d}.npy',q.astype(np.float32))
            entries.append((p.reshape(-1,3),q.reshape(-1,3),y.reshape(-1,3)))
        report[stop]={'weights':w.tolist(),'result':prior.measure(entries)}
        (OUT/'results.json').write_text(json.dumps(report,indent=2))
        print('score',stop,report[stop],flush=True)


if __name__=='__main__':main()
