"""Chronological reference for the stock-regularized common-market Y3 readout."""
import json
import numpy as np
import experiment_dense_market_slow_prior_v1 as market
from experiment_dense_session_slope_v1 import GEO

parent=market.parent
base=market.base
OUT=base.ROOT/"artifacts/dense_market_reliability_tape_v1"


def value(stock,common):
    g,s=np.mean(common,0),np.mean(stock,0)
    n=(len(g)-1)//3
    sd=np.sqrt(g.diagonal()[:n]).clip(.006);ss=np.sqrt(s.diagonal()[:n]).clip(.03)
    ratio=sd/ss
    c=.5*(g[:n,:n]+s[:n,:n]*np.outer(ratio,ratio))
    payoff=.5*(g[:n,-1]+s[:n,-1]*ratio*np.sqrt(g[-1,-1]/s[-1,-1]))
    return np.linalg.solve(c+.25*np.diag(sd**2),payoff)


def observation(raw,geo):
    k,co,row=geo
    valid=~np.all(raw==row,axis=2)
    price=np.log1p(k*(raw[:,:,0].astype(float)-.00136))*co[2,1]/k
    return (price*valid).sum(1)/valid.sum(1).clip(1)


def make(raw,grid,p,code,sb,mb,pw,mw,now,geo):
    x,av=parent.state(raw,grid,code,with_market=True)
    m=market.market_state(av,grid)
    z=x@sb
    cm=m@mb
    means=p.mean(1,dtype=float)
    d=np.column_stack((means,cm))@mw
    old=p.copy()
    old[:,:,2]+=np.concatenate((p-p.mean(1,keepdims=True),z),2)@pw
    old[:,:,2]+=d[:,None]
    return dict(reference=old,change=d,directions=cm,now=m@now,cash=observation(raw,geo),means=means)


def main():
    OUT.mkdir(exist_ok=True)
    code=parent.codec();stock=[];common=[];sfit=[];mfit=[]
    with np.load(GEO) as f:geo=float(f["k"]),f["coeff"],f["row"]
    for d in range(16):
        load=lambda k:np.load(base.CACHE/f"day_{d:02d}_{k}.npy",mmap_mode="r")
        raw,grid,y=load("x"),load("time").astype(int),load("y")
        x,av=parent.state(raw,grid,code,with_market=True)
        m=market.market_state(av,grid)
        if d>=4:
            if d%4==0:
                sb=parent.components(stock);mb=market.operators(stock,common);now=value(stock,common)
            p=np.load(market.TAPE/f"forecast_{d:02d}.npy")
            means=p.mean(1,dtype=float)
            z=x@sb;cm=m@mb
            if d>=6:
                if d==6 or d%4==0:pw=parent.fit(sfit,0);mw=market.fit(mfit,2)
                change=np.column_stack((means,cm))@mw
                old=p.copy()
                old[:,:,2]+=np.concatenate((p-p.mean(1,keepdims=True),z),2)@pw
                old[:,:,2]+=change[:,None]
                np.savez(OUT/f"day_{d:02d}.npz",reference=old,change=change,directions=cm,now=m@now,cash=observation(raw,geo),means=means)
            sfit.append(parent.stats(p,y,z));mfit.append(market.stats(p,y,cm))
        stock.append(parent.moment(x,grid,y));common.append(market.record(m,grid,y))
    with np.load(market.OUT/"weights.npz") as f:
        np.testing.assert_array_equal(market.operators(stock,common),f["bank"])
        np.testing.assert_array_equal(market.fit(mfit,2),f["w"])
    np.save(OUT/"now.npy",value(stock,common))
    print("development tape rebuilt and final market coefficients verified",flush=True)


def outer(d):
    load=lambda k:np.load(base.CACHE/f"day_{d:02d}_{k}.npy",mmap_mode="r")
    raw,grid=load("x"),load("time").astype(int)
    p=market.released(d,raw)
    with np.load(parent.OUT/"weights.npz") as f,np.load(market.OUT/"weights.npz") as w,np.load(GEO) as g:
        args=f["code"],f["bank"],w["bank"],f["w"],w["w"],np.load(OUT/"now.npy"),(float(g["k"]),g["coeff"],g["row"])
        result=make(raw,grid,p,*args)
        if d==16:
            n=2473
            short=make(raw[:n],grid[:n],p[:n],*args)
            for k in short:np.testing.assert_allclose(short[k],result[k][:n],atol=2e-7,rtol=0)
    return grid,result,load("y")


if __name__=="__main__":main()
