"""Combine a factor market forecast with zero-mean stock-specific neural corrections."""
import json
import numpy as np
import experiment_factor_exposure_return_v1 as ex
import experiment_pooled_price_impact_v1 as impact
import experiment_multiscale_return_net_v1 as v1
import experiment_causal_path_tcn_v1 as tcn
import experiment_relative_price_endpoint_v1 as endpoint
prior=v1.prior
OUT=prior.ART/'separate_market_relative_v1'


def relative(x):
    return x-x.mean(1,keepdims=True)


def optimize(rows):
    p,y,*s=map(np.concatenate,zip(*rows))
    x=np.stack((p,*s),axis=-1).reshape(-1,3,len(s)+1).astype(float)
    x-=x.mean(0);y=y.reshape(-1,3).astype(float);y-=y.mean(0)
    gram=np.einsum('nki,nkj->kij',x,x);rhs=np.einsum('nki,nk->ki',x,y)
    var=(y*y).sum(0)
    import itertools
    best=np.full(3,-np.inf);weights=np.zeros((len(s),3))
    for trial in itertools.product((0.,.5,1.),(0.,.5,1.),(0.,.25,.5)):
        w=np.r_[1.,trial]
        score=(rhs@w)/np.sqrt(var*np.einsum('i,kij,j->k',w,gram,w))
        score-=.00035*np.sum((w[1:]-.5)**2)
        ix=score>best;weights[:,ix]=np.array(trial)[:,None];best=np.maximum(best,score)
    return weights


def main():
    OUT.mkdir(exist_ok=True);early=[];report={}
    w_impact=np.array(json.loads((impact.OUT/'results.json').read_text())['16']['weights'])
    for end,stop in [(12,14),(14,16),(16,20)]:
        geo=ex.geometry(end);w_exp=ex.fit(range(8,end),geo)
        rows=[]
        for d in range(end,stop):
            p=ex.v1_base(d);y=prior.labels(d).reshape(p.shape)
            side=np.load(impact.OUT/f'side_{d}.npy')
            base=v1.blend(p,side,w_impact)
            a=relative(np.load(tcn.OUT/f'prediction_{d}.npy')-p)
            b=relative(np.load(endpoint.OUT/f'pred_{d}.npy')-p)
            s=ex.features(ex.grid(d),p,geo)@w_exp
            rows.append((base.astype(np.float32),y,a,b,s.astype(np.float32)))
        if end<16:
            early.extend(rows);weights=optimize(early)
        entries=[]
        for d,(p,y,*terms) in zip(range(end,stop),rows):
            q=p+sum(w*z for w,z in zip(weights,terms))
            np.testing.assert_allclose(q.mean(1),p.mean(1),atol=3e-6)
            np.save(OUT/f'pred_{d}.npy',q.astype(np.float32))
            entries.append((p.reshape(-1,3),q.reshape(-1,3),y.reshape(-1,3)))
        report[stop]={'weights':weights.tolist(),'result':prior.measure(entries)}
        print('score',stop,report[stop],flush=True)
        (OUT/'results.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
