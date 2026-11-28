"""Estimate which parts of the quote-revision/return relation replicate for individual stock identities."""
import numpy as np
import experiment_dense_refresh_payoff_instruments_v1 as own
from prepare_dense_refresh_payoff_reference_v1 import OUT as REFERENCE

onset,rel,book=own.onset,own.rel,own.book
OUT=REFERENCE.parent/"dense_stock_specific_refresh_response_v1"
ARMS=("stock_specific_revision","stock_specific_payoff","both_stock_effects")
C,D=own.C,own.D
N=C+2*D+3
X,A=slice(C,C+D),slice(C+D,C+2*D)


def collect():
    OUT.mkdir(exist_ok=True)
    for d in range(4,16):
        raw,grid,y=[onset.settled.load(d,k) for k in ("x","time","y")]
        p=(np.load(REFERENCE/f"rolling_{d:02d}.npy") if d>=6 else
           onset.settled.head.ranks.band.pairs.net.future.forecast(d))
        cur=p-p.mean(1,keepdims=True);label=y-y.mean(1,keepdims=True)
        slope=((cur*label).sum((0,1))/(cur*cur).sum((0,1))).clip(.3,3)
        err=label/slope-cur
        g=np.zeros((2,50,N,N));n=np.zeros(2);mean=np.zeros((2,50,N))
        for t,x,a in own.panels(raw,grid,p,6,d%6,True):
            z=np.concatenate((x,a,err[t]),axis=2).transpose(1,0,2)
            for fold in (0,1):
                use=((t//240)%2==fold)&(t%240<180)
                block=z[:,use]
                g[fold]+=block.swapaxes(1,2)@block
                mean[fold]+=block.sum(1);n[fold]+=use.sum()
        g/=n[:,None,None,None];mean/=n[:,None,None]
        np.save(OUT/f"gram_{d:02d}.npy",g-mean[:,:,:,None]*mean[:,:,None,:])


def reliability(days,left,right):
    # Two disjoint sets of 240-tick blocks supply a conservative prior when
    # there is only one date in the fitting half. Their labels do not overlap.
    if len(days)==1:
        replicas=days[0]
        cap=.25
    else:
        pooled=days.mean(1)
        replicas=[pooled[j::2].mean(0) for j in (0,1)]
        cap=.85
    full=days.mean((0,1,2))
    sd=np.sqrt(full.diagonal().clip(1e-7))
    scale=np.outer(sd[left],sd[right])
    deviations=[]
    for sample in replicas:
        cross=sample[:,left,right]/scale
        deviations.append(cross-cross.mean(0))
    a,b=deviations
    return np.clip(2*(a*b).sum()/((a*a+b*b).sum()+1e-10),0,cap)


def local_model(rows,keep,stage):
    m=rows.mean((0,1))
    cov=keep*m+(1-keep)*m.mean(0)
    result=[]
    for g in cov:
        tr=np.eye(len(g));tr[:C,C:C+2*D]=-own.solve(g[:C,:C],g[:C,C:C+2*D],.08)
        clean=tr.T@g@tr
        if stage:
            result.append(tr[:C+D,X]@own.solve(clean[X,X],clean[X,A],.8))
        else:
            result.append(own.solve(clean[A,A],clean[A,-3:],.8))
    return np.stack(result)


def train(end):
    days=np.stack([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)])
    w=np.zeros((50,C+D,3,3))
    for parity in (0,1):
        first,second=[days[k::2] for k in (parity,1-parity)]
        stage_r=reliability(first,X,A)
        payoff_r=reliability(second,A,slice(-3,None))
        common_stage=local_model(first,0.,True)
        common_pay=local_model(second,0.,False)
        adapted_stage=local_model(first,stage_r,True)
        adapted_pay=local_model(second,payoff_r,False)
        for j,(f,b) in enumerate(((adapted_stage,common_pay),(common_stage,adapted_pay),(adapted_stage,adapted_pay))):
            # Isolate the contribution of stock-specific coefficients.
            w[:,:,j]+=.5*(f@b-common_stage@common_pay)
    return w


def design(raw,grid,code,p,bank):
    out=np.zeros((*p.shape[:2],3,3))
    for t,x,_ in own.panels(raw,grid,p,1,0,False):
        z=np.einsum("tsf,sfah->tsah",x,bank)
        out[t]=z-z.mean(1,keepdims=True)
    return out


def predict(raw,grid,p,params):
    return book.path.corrected(p,design(raw,grid,params["code"],p,params["bank"]),params)


if __name__=="__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
