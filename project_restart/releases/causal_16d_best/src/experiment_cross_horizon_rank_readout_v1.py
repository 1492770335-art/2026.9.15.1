"""Transfer cross-sectional ranking information across horizons while projecting out existing stock forecasts."""
import json
import numpy as np
import experiment_phase_future_book_v1 as phase
import experiment_orthogonal_expert_innovations_v1 as orth
from experiment_causal_cross_horizon_ridge_v1 import measure

OUT=phase.inn.prior.ART/'cross_horizon_rank_readout_v1'
SOURCES=('phase_future_book_v1','symmetric_pressure_network_v1','pairwise_spread_decoder_v1','shared_bar_gru_v1')

def features(p,models):
    x=p.astype(float)-p.mean(1,keepdims=True)
    gram=x.transpose(0,2,1)@x/50
    regularizer=np.eye(3)[None]*(.028*gram.diagonal(axis1=-1,axis2=-2).clip(.006,None)+.0006)[:,None,:]
    cols=[]
    for n in models:
        r=n.astype(float)-n.mean(1,keepdims=True)
        coefficients=np.linalg.solve(gram+regularizer,x.transpose(0,2,1)@r/50)
        extra=r-x@coefficients
        # Preserve cross-horizon information about which stocks have moved up or down in rank.
        cols.append(extra)
    return np.concatenate(cols,axis=2).astype('float32')

def stats(z,p,y):
    a=z.reshape(-1,z.shape[-1]).astype(float)
    b=p.reshape(-1,3).astype(float);b-=b.mean(0);v=y.reshape(-1,3).astype(float);v-=v.mean(0)
    slope=((b*v).sum(0)/(b*b).sum(0)).clip(.4,2.5)
    return a.T@a/len(a),a.T@(v/slope-b)/len(a)

def fit(hist):
    g,b=(np.mean(items,axis=0) for items in zip(*hist))
    reg=1.55*np.diag(g.diagonal().clip(.0003,None))
    ds=np.stack([np.linalg.solve(a+reg,r) for a,r in hist])
    consistency=abs(np.sign(ds).mean(0))
    return (.52*(.55*np.linalg.solve(g+reg,b)+.45*np.median(ds,0))*consistency).clip(-.14,.18)

def main():
    OUT.mkdir(exist_ok=True)
    hist=[];report={};eligible=np.ones(3,dtype=bool)
    for d in range(12,20):
        if d in (14,16):
            if d==16 and not eligible.any():break
            coef=fit(hist)*eligible;rows=[]
            if d==16:np.save(OUT/'coef.npy',coef)
        root=phase.OUT if d<14 else orth.OUT
        p=np.load(root/f'pred_{d}.npy');y=phase.inn.prior.labels(d).reshape(p.shape)
        sources=[np.load(phase.inn.prior.ART/name/f'side_{d}.npy') for name in SOURCES]
        z=features(p,sources)
        if d>=14:
            q=(p+z@coef).astype('float32')
            if d==16:np.testing.assert_allclose(q[:155],(p[:155]+features(p[:155],[s[:155] for s in sources])@coef).astype('float32'),atol=1e-6)
            rows.append(tuple(a.reshape(-1,3) for a in (p,q,y)))
            np.save(OUT/f'pred_{d}.npy',q)
        if d<16:hist.append(stats(z,p,y))
        if d in (15,19):
            val=measure(rows);report[str(d+1)]=val
            if d==15:
                day_gains=[np.array(measure([row])['candidate'])-measure([row])['base'] for row in rows]
                eligible=np.min(day_gains,axis=0)>0
            (OUT/'results.json').write_text(json.dumps(report,indent=2));print(d+1,val,eligible,flush=True)
if __name__=='__main__':main()
