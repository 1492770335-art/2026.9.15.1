"""Forecast corrections from historical predictions of a price endpoint and peer price changes."""
import json
import numpy as np
import experiment_stock_calibration_v1 as base
from experiment_causal_return_basket_v1 import grid, moving_sum, basket
from experiment_multiscale_return_net_v1 import SCALE
prior = base.prior
OUT = prior.ART / 'endpoint_innovation_v1'
H = np.array([5,20,60])

def forecast(d):
    if d < 12:
        return base.forecasts(d)
    return np.load(base.OUT / f'pred_{d}.npy').astype(float)

def designs(raw,p,graph):
    t=np.arange(len(p));code=np.arange(50)
    r=raw[:,:,0].astype(float)-.00136
    price=r.cumsum(0);peer=price@graph.T
    last=np.maximum.accumulate(np.where(abs(r)>.018,t[:,None],0),axis=0)
    age=t[:,None]-last
    drift=peer-peer[last,code]
    local=[np.arcsinh(drift),np.arcsinh(drift)/(1+age/12)]
    market=[]
    realized=np.sqrt((8+moving_sum(r*r,96))/(8+np.minimum(t+1,96))[:,None]).clip(.45,3.5)
    def add(value):
        value=value.clip(-12,12)
        m=value.mean(1)
        local.append((value-m[:,None]).astype(np.float32))
        market.append(m)
    for j,h in enumerate(H):
        for k in sorted(set([1,3,8,16,32,60,int(h)])):
            idx=np.maximum(0,t-k)
            # A past prediction has matured partly; subtract movement realized up to now.
            path=price-price[idx]
            old=p[idx,:,j]/SCALE[j]
            elapsed=np.minimum(t,k)[:,None]
            expected_fraction=np.minimum(1,elapsed/h)
            surprise=(path-expected_fraction*old)/np.sqrt(1+elapsed)
            revise=(p[:,:,j]-p[idx,:,j])
            add(surprise);add(revise)
            if k in (3,16,60):
                add(surprise*realized);add(surprise/(.5+realized))
                add((surprise@graph.T-surprise)*np.minimum(1,age/8))
        lastp=p[last,code,j]
        add((p[:,:,j]-lastp))
        add(p[:,:,j]*np.minimum(1.,age/8))
        for m in range(j+1,3):
            # The old and current forecasts refer to exactly the same terminal tick.
            gap=H[m]-h; idx=np.maximum(0,t-gap)
            aligned=SCALE[j]*(p[idx,:,m]/SCALE[m]-(price-price[idx]))-p[:,:,j]
            aligned[:gap]=0
            add(aligned)
            add(aligned*np.minimum(1.,age/8))
            add(aligned@graph.T)
    local=np.stack(local,axis=-1)
    local-=local.mean(1,keepdims=True)
    return np.column_stack(market),local.astype(np.float32)

def fit(rows,ridge):
    gs,bs=zip(*rows)
    g=np.mean(gs,0); b=np.mean(bs,0)
    penalty=ridge*np.diag(np.maximum(1e-6,g.diagonal()))
    stable=np.median([np.linalg.solve(a+penalty,v) for a,v in rows],axis=0)
    return .7*np.linalg.solve(g+penalty,b)+.3*stable

def train(end,graph):
    markets=[]; relatives=[]
    for d in range(8,end):
        p=forecast(d);error=prior.labels(d).reshape(p.shape)-p
        m,z=designs(grid(d),p,graph); em=error.mean(1)
        markets.append((m.T@m/len(m),m.T@em/len(m)))
        z=z[d%4::4].reshape(-1,z.shape[-1]).astype(float)
        y=(error-em[:,None])[d%4::4].reshape(-1,3)
        relatives.append((z.T@z/len(z),z.T@y/len(z)))
    return fit(markets,.9),fit(relatives,.35)

def strength(rows):
    gram=0;rhs=0;ysq=0;N=0;sumx=0;sumy=0
    for p,y,m,r in rows:
        x=np.stack((p,np.broadcast_to(m[:,None],p.shape),r),axis=-1).reshape(-1,3,3).astype(float)
        y=y.reshape(-1,3).astype(float);N+=len(y)
        gram+=np.einsum('nki,nkj->kij',x,x);rhs+=np.einsum('nki,nk->ki',x,y);ysq+=(y*y).sum(0)
        sumx+=x.sum(0);sumy+=y.sum(0)
    gram-=sumx[:,:,None]*sumx[:,None,:]/N;rhs-=sumx*sumy[:,None]/N;ysq-=sumy**2/N
    best=np.full(3,-np.inf);weights=np.zeros((2,3))
    for a in [0,.25,.5,.85,1.2]:
        for b in [0,.25,.5,.85,1.2]:
            w=np.array([1,a,b]);c=(rhs@w)/np.sqrt(ysq*(w@gram@w))-.00015*((a-.4)**2+(b-.4)**2)
            ix=c>best;weights[:,ix]=[[a],[b]];best=np.maximum(best,c)
    return weights

def main():
    OUT.mkdir(exist_ok=True);early=[];report={}
    for end,stop in [(12,14),(14,16),(16,20)]:
        graph=basket(end);cm,cr=train(end,graph);rows=[]
        for d in range(end,stop):
            p=forecast(d);raw=grid(d);x,z=designs(raw,p,graph)
            if d==end:
                xx,zz=designs(raw[:214],p[:214],graph)
                np.testing.assert_allclose(xx,x[:214],atol=1e-7)
                np.testing.assert_allclose(zz,z[:214],atol=1e-5)
            rows.append((p,prior.labels(d).reshape(p.shape),x@cm,z@cr))
        if end<16:
            early.extend(rows);weights=strength(early)
        else:np.savez(OUT/'model.npz',graph=graph,cm=cm,cr=cr,weights=weights)
        entries=[]
        for d,(p,y,m,r) in zip(range(end,stop),rows):
            q=p+m[:,None]*weights[0]+r*weights[1]
            np.save(OUT/f'pred_{d}.npy',q.astype(np.float32))
            entries.append(tuple(x.reshape(-1,3) for x in (p,q,y)))
        report[stop]={'weights':weights.tolist(),'result':prior.measure(entries)}
        (OUT/'results.json').write_text(json.dumps(report,indent=2));print(stop,report[stop],flush=True)

if __name__=='__main__':main()
