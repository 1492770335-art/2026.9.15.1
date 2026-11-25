"""Keep the overshoot definition fixed when permuting the identity or age of its one-tick expectation."""
import json
import numpy as np

import experiment_y3_market_reversal_flow_v1 as market

trial=market.trial


def alternatives(raw,z):
    r=raw[:,:,0].astype('float32')-.00136
    exp=market.prior.predicted(raw,market.prior.teacher())
    delayed=np.zeros_like(exp);delayed[16:]=exp[:-16]
    variants=[z]
    for other in (np.roll(exp,17,axis=1),delayed):
        extra=np.where(r*other>0,np.sign(r)*np.maximum(abs(r)-abs(other),0),0.)
        panels=[]
        for i,features in enumerate(z):
            a=extra[:,i::2]
            v=np.stack((np.tanh(a).mean(1),np.tanh(4*a).mean(1)),1)
            cs=np.vstack((np.zeros(2),v.cumsum(0,dtype=float)))
            end=np.arange(len(raw))+1
            source=features.copy()
            for j,width in enumerate((4,16,48)):
                start=(end-width).clip(0)
                source[:,18+2*j:20+2*j]=(cs[end]-cs[start])/(end-start)[:,None]
            panels.append(source)
        variants.append(panels)
    return variants


def main():
    selected=json.loads((market.OUT/'results.json').read_text())['selected']
    assert market.ARMS[selected]=='overshooting moves'
    rows=[[],[],[]];byday=[]
    for d in range(4,16):
        raw=trial.clock.book.rawday(d)
        y=trial.clock.labels(d).reshape(-1,50,3)
        ref=[market.prior.path.early.load(d,raw)] if d<8 else market.history.references(d,raw)
        z=alternatives(raw,market.pools(raw,market.prior.teacher()))
        if d>=8:
            if d%2==0:
                heads=[market.fit(h) for h in rows]
            p=ref[0]
            delta=[market.changes(x,p,h)[selected] for x,h in zip(z,heads)]
            delta.extend(market.changes(x,p,heads[0])[selected] for x in z[1:])
            byday.append(trial.scoring.audit.blockstats(p,y,delta).sum(0))
        for h,x in zip(rows,z):
            records=[market.record(x,p,y) for p in ref]
            h.append([np.mean(v,axis=0) for v in zip(*records)])
    report=dict(days=list(range(9,17)),variants=['true','permuted refitted','delayed refitted','permuted frozen','delayed frozen'],
                y3_gains=trial.scoring.audit.pooled(np.sum(byday,0)).tolist())
    (market.OUT/'matched_overshoot_placebo.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)

if __name__=='__main__':main()
