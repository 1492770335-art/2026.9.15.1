"""Calibrate best bid and ask sizes against same-quote executions using only the first four dates."""
from functools import lru_cache
import json

import numpy as np
from scipy.optimize import least_squares

import audit_y3_top_queue_scale as scale

trial, tape = scale.trial, scale.field.tape
OUT = scale.OUT


def volumes(z, params):
    shift, deep, logscale, lower, upper = params
    total=np.exp((deep*z[...,1]+logscale).clip(-20,12))*(z[...,0]-shift).clip(.00001)
    bid_share=((z[...,2]-lower)/(upper-lower)).clip(0,1)
    return total*bid_share,total*(1-bid_share)


def events(raw, state):
    mid,half,off,avg,valid,frac,buy,sell=tape.prices(raw,state)
    prev=np.r_[0,np.arange(len(raw)-1)]
    s=(buy>0).astype(int)-(sell>0).astype(int)
    expected=mid[prev]+s*half[prev]
    fixed=(mid==mid[prev])&(half==half[prev])
    eligible=valid&valid[prev]&fixed&(s!=0)&(mid+off==expected)&(abs(mid+avg-expected)<.075)
    eligible[0]=False
    at,code=np.nonzero(eligible)
    def view(t):
        return raw[t,code][:,[45,37,21]].astype(float)
    return view(at-1),view(at),s[at,code],(buy+sell)[at,code],code


def errors(record, params):
    prev,now,sign,volume,_=record
    b0,a0=volumes(prev,params)
    b1,a1=volumes(now,params)
    hit=np.where(sign>0,a0-a1,b0-b1)
    opposite=np.where(sign>0,b0-b1,a0-a1)
    denom=volume.clip(.006)
    return np.column_stack(((hit-volume)/denom,opposite/denom))


@lru_cache(maxsize=1)
def calibration():
    state=tape.calibration()
    lo,hi=scale.limits()
    init=np.array([-1.1346,1.04965,0.,lo,hi])
    records=[events(trial.clock.book.rawday(d),state) for d in range(4)]
    z=[np.concatenate(col) for col in zip(*records)]
    # Select transactions consistent with an approximately isolated depletion in both queues.
    # Refine price-independent coefficients on a fixed subsample of the early dates.
    ids=np.arange(1,len(z[2]),3)
    z=[v[ids] for v in z]
    for tolerance in (.22,.075,.035):
        residual=errors(z,init)
        mask=(np.max(abs(residual),axis=1)<tolerance)&(z[3]>.012)
        trusted=[v[mask] for v in z]
        init=least_squares(lambda w:errors(trusted,w).ravel(),init,
                           bounds=([-1.8,.6,-.8,lo-.07,hi-.05],[-.9,1.5,.8,lo+.05,hi+.07]),
                           loss='soft_l1',f_scale=.004).x
    return dict(**state, touch=init)


def size(raw,state):
    return volumes(raw[:,:,[45,37,21]].astype(float),state['touch'])


def validate():
    OUT.mkdir(exist_ok=True)
    state=calibration()
    rows=[];oldfills=[]
    for d in range(8,16):
        rec=events(trial.clock.book.rawday(d),state)
        e=errors(rec,state['touch'])
        v=np.minimum(rec[3],.006)/.006
        joint=np.max(abs(e),axis=1)<.10
        # Swapping the aggressor exchanges the depleted and untouched sides.
        reverse=np.max(abs(np.column_stack((e[:,1]-v,e[:,0]+v))),axis=1)<.10
        rows.append([len(e),int(joint.sum()),int(reverse.sum())])
        x,_,_,_,side,amount,_,_=scale.field.samples(d)
        b,a=volumes(x[:,[45,37,21]],state['touch'])
        expected=np.where(side>0,a,b)
        oldfills.append(np.stack((amount,expected),axis=1))
    amount,estimated=np.concatenate(oldfills).T
    ok=amount>.006
    err=abs(estimated[ok]/amount[ok]-1)
    data=np.sum(rows,axis=0)
    report=dict(discovery=[1,4],replication=[9,16],coefficients=state['touch'].tolist(),
                isolated_fill_candidates=int(data[0]),conservation_hits=int(data[1]),reverse_hits=int(data[2]),
                joint_fraction=float(data[1]/data[0]),reversed_fraction=float(data[2]/data[0]),
                independent_two_price_events=int(ok.sum()),two_price_within_10pct=float(np.mean(err<.1)),
                two_price_median_relative_error=float(np.median(err)),daily=rows)
    (OUT/'touch_sizes.json').write_text(json.dumps(report,indent=2))
    print(report,flush=True)


if __name__=='__main__':
    validate()
