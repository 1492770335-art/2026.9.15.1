"""Measure which input fields respond differently to movements on the two inferred price sides."""
import json
from functools import lru_cache

import numpy as np

import y3_inferred_quote_levels as quote

trial = quote.trial
OUT = quote.grid.grid.history.incumbent.OUT


def profile(day):
    raw = trial.clock.book.rawday(day)
    mid,half,_,good = quote.infer(raw)
    dmid,dw = np.diff(mid,axis=0),np.diff(half,axis=0)
    bid,ask = dmid-dw,dmid+dw
    good = good[1:]&good[:-1]&(abs(dmid)<40)
    left=good&(bid!=0)&(ask==0)
    right=good&(bid==0)&(ask!=0)
    # Within each stock pair price changes of equal direction and comparable size.
    parts=[]
    for sign in (-1,1):
        for l,u in ((1,2),(3,6),(7,99)):
            a=left & (sign*bid>=l)&(sign*bid<=u)
            b=right & (sign*ask>=l)&(sign*ask<=u)
            n,m=a.sum(0),b.sum(0)
            both=np.flatnonzero((n>=5)&(m>=5))
            if not len(both):
                continue
            s=[]
            for stock in both:
                x=np.arcsinh(raw[:,stock,:83].astype('float32'))
                delta=np.abs(np.diff(x,axis=0))
                first,second=delta[a[:,stock]].mean(0),delta[b[:,stock]].mean(0)
                s.append((first-second)/(first+second+.005))
            parts.append(np.mean(s,0))
    return np.mean(parts,axis=0)


@lru_cache(maxsize=1)
def training_profile():
    return np.mean([profile(d) for d in range(4)],axis=0)


def run():
    a=training_profile()
    b=np.mean([profile(d) for d in range(8,16)],axis=0)
    order=np.argsort(-abs(a))
    rows=[dict(field=int(i),discovery=float(a[i]),replication=float(b[i])) for i in order]
    (OUT/'quote_side_fields.json').write_text(json.dumps(dict(discovery=[1,4],replication=[9,16],fields=rows),indent=2))
    print([(r['field'],round(r['discovery'],2),round(r['replication'],2)) for r in rows[:30]],flush=True)


if __name__=='__main__':
    run()
