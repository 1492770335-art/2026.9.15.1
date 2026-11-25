"""Calibrate the compounded price-label relationship without using any validation date to fit it."""
import json
import numpy as np
from scipy.optimize import minimize_scalar
import experiment_cutoff_transition_rerun_v1 as incumbent
from experiment_causal_cross_horizon_ridge_v1 import ART,labels

OUT=ART/'compounded_return_audit_v1'
H=(5,20,60)
book=incumbent.walk.book


def template():
    samples=np.concatenate([book.rawday(d)[d%21::21].reshape(-1,100) for d in range(8)])
    rows,count=np.unique(samples,axis=0,return_counts=True)
    return rows[count.argmax()]


def present(raw,row):
    return ~np.all(raw==row,axis=2)


def compound(raw,k):
    r=raw[:,:,0].astype(float)-.00136
    return np.vstack((np.zeros_like(r[:1]),np.log1p(k*r).cumsum(0)))


def span(log_price,t,h,k):
    return np.expm1(log_price[t+h+1]-log_price[t+1])/k


def geometry(row):
    rows=[]
    for d in range(4):
        raw=book.rawday(d)
        valid=present(raw,row)
        miss=np.vstack((np.zeros((1,50),int),(~valid).cumsum(0)))
        y=labels(d).reshape(-1,50,3)
        t=np.arange(d%13,len(raw)-60,13)
        good=(miss[t+61]-miss[t]==0)
        rows.append((raw,t,y[t],good))
    def estimate(k):
        xs=[[] for _ in H];ys=[[] for _ in H]
        for raw,t,y,ok in rows:
            price=compound(raw,k)
            for j,h in enumerate(H):
                xs[j].append(span(price,t,h,k)[ok]);ys[j].append(y[:,:,j][ok])
        coef=[];loss=0.
        for a,b in zip(xs,ys):
            x=np.concatenate(a);y=np.concatenate(b)
            use=abs(x)<65
            mat=np.column_stack((np.ones(use.sum()),x[use]))
            w=np.linalg.lstsq(mat,y[use],rcond=None)[0]
            e=y[use]-mat@w
            loss+=np.minimum(e**2,.008**2).mean()
            coef.append(w)
        return loss,np.array(coef)
    k=minimize_scalar(lambda v:estimate(v)[0],bounds=(.0003,.0013),method='bounded',options={'xatol':1e-9}).x
    coeff=estimate(k)[1]
    return k,coeff


def main():
    OUT.mkdir(exist_ok=True)
    row=template();k,w=geometry(row)
    days=[]
    for d in range(16):
        raw=book.rawday(d);y=labels(d).reshape(-1,50,3)
        valid=present(raw,row)
        miss=np.vstack((np.zeros((1,50),int),(~valid).cumsum(0)))
        price=compound(raw,k);audit=[]
        for j,h in enumerate(H):
            t=np.arange(len(y)-h)
            inside=miss[t+h+1]-miss[t]>0
            error=y[t,:,j]-(w[j,0]+w[j,1]*span(price,t,h,k))
            big=abs(error)>.15
            audit.append(dict(clean_count=int((~inside).sum()),clean_rmse=float(np.sqrt((error[~inside]**2).mean())),
                              flagged_count=int(inside.sum()),large_error_clean=int((big&~inside).sum()),
                              large_error_flagged=int((big&inside).sum())))
        days.append(dict(day=d+1,template_rows=int((~valid).sum()),statistics=audit))
    np.savez(OUT/'model.npz',row=row,k=k,coeff=w)
    (OUT/'results.json').write_text(json.dumps(dict(k=float(k),coeff=w.tolist(),parameter_fit_days=[1,4],
        template_fit_days=[1,8],audit=days),indent=2))
    print(k,w.tolist(),flush=True)

if __name__=='__main__':main()
