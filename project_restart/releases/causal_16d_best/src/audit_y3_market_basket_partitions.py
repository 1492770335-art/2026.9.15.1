"""Repeat the frozen two-basket confidence adjustment on eight alternative stock partitions in training dates."""
import json
import numpy as np

import experiment_y3_overshoot_basket_agreement_v1 as gate

market,history,trial=gate.market,gate.history,gate.trial


def evidences(raw,params,perms):
    r=raw[:,:,0].astype('float32')-.00136
    pred=market.prior.predicted(raw,params['teacher'])
    extra=np.where(r*pred>0,np.sign(r)*np.maximum(abs(r)-abs(pred),0.),0.)
    values=np.stack((np.tanh(extra),np.tanh(4*extra)),2)
    coef=params['w'][21:27,int(params['pick'])]
    result=[]
    for perm in perms:
        pair=[]
        for side in (0,1):
            panel=values[:,perm[side::2]].mean(1)
            cs=np.vstack((np.zeros(2),panel.cumsum(0,dtype=float)))
            end=np.arange(len(raw))+1
            spans=[]
            for width in (4,16,48):
                start=(end-width).clip(0)
                spans.append((cs[end]-cs[start])/(end-start)[:,None])
            pair.append(np.column_stack(spans).astype('float32')@coef)
        a,b=pair
        result.append(2*a*b/(a*a+b*b+1e-8))
    return result


def main():
    saved=json.loads((market.OUT/'results.json').read_text())
    rng=np.random.default_rng(27031)
    perms=[np.arange(50)]+[rng.permutation(50) for _ in range(8)]
    rows=[];grams=[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d);y=trial.clock.labels(d).reshape(-1,50,3)
        z=market.pools(raw,market.prior.teacher())
        base=[market.prior.path.early.load(d,raw)] if d<8 else history.previous.references(d,raw)
        if d>=8:
            if d%2==0:params=dict(market.fit(rows),pick=saved['selected'])
            p=history.references(d,raw)[0]
            common,changes=gate.signal(p,z,params)
            strength=evidences(raw,params,perms)
            corrections=[.4*common*v for v in strength]
            np.testing.assert_allclose(corrections[0],changes[3],rtol=0,atol=1e-6)
            corrections.append(np.mean(corrections,axis=0))
            broadcast=[np.broadcast_to(v[:,None],p.shape[:2]) for v in corrections]
            grams.append(trial.scoring.audit.blockstats(p,y,broadcast).sum(0))
        records=[market.record(z,p,y) for p in base]
        rows.append([np.mean(v,axis=0) for v in zip(*records)])
    out=trial.scoring.audit.pooled(np.sum(grams,axis=0))
    report=dict(days=list(range(9,17)),reference=out[0],alternatives=out[1:-1].tolist(),averaged=out[-1])
    (gate.OUT/'training_partitions.json').write_text(json.dumps(report,indent=2))
    print(report)

if __name__=='__main__':main()
