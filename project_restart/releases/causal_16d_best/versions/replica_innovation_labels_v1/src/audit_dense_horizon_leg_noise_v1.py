"""Attribute the incumbent's sixty-step price error to disjoint legs on training-period forecast days."""
import json
import numpy as np

from experiment_dense_market_conditional_graph_risk_v1 import event,onset,rel,strip
import experiment_dense_forecast_error_passages_v1 as curve

OUT = curve.OUT.parent/'dense_leg_news_readout_v1'
STOPS = (5,20,60)


def physical(p,code):
    k,offset,scale = code[0],code[1:4],code[4:]
    return np.log1p(((p-offset)*k/scale).clip(-.99))*scale[-1]/k


def day_moments(day):
    raw,grid = [onset.settled.load(day,k) for k in ('x','time')]
    p = curve.read_p(day)
    data = [[] for _ in range(2)]
    for a,b,m,ok in event.sessions(raw,grid):
        price = m.cumsum(0)
        gap = np.vstack((np.zeros(50,int),(~ok).cumsum(0)))
        t = np.arange(day%5,b-a-60,5)
        quotes = physical(p[a:b],curve.codec())
        forecasts = np.diff(quotes[t],axis=2,prepend=np.zeros_like(quotes[t,:,:1]))
        actual = np.stack([price[t+v]-price[t+u] for u,v in zip((0,5,20),STOPS)],axis=-1)
        valid = gap[t+61]==gap[t]
        cash = actual-forecasts
        # Sum of the short-horizon surprises after issue on the same realised path.
        short = []
        for u,v in zip((0,5,20),STOPS):
            marks = sum(quotes[t+i,:,0] for i in range(u,v,5))
            short.append(actual[:,:,len(short)]-marks)
        x = np.concatenate((cash,np.stack(short,axis=2)),axis=2)
        mean = (x*valid[...,None]).sum(1)/valid.sum(1,keepdims=True).clip(1)
        data[0].append(mean[valid.sum(1)>=35])
        data[1].append((x-mean[:,None])[valid])
    return np.stack([np.cov(np.concatenate(z).T,bias=True) for z in data])


def diagnose(g):
    leg = g[:3,:3]
    total = leg.sum()
    share = leg.sum(1)/total
    noise = np.diag(g[:3,3:])/np.sqrt(leg.diagonal()*g[3:,3:].diagonal()).clip(1e-9)
    return dict(share_of_total_error=share.tolist(),corr_with_later_short_surprises=noise.tolist())


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    report = {}
    for start in (8,12):
        g = np.mean([day_moments(d) for d in range(start,start+4)],axis=0)
        report[start+1] = dict(common=diagnose(g[0]),relative=diagnose(g[1]))
    (OUT/'error_attribution.json').write_text(json.dumps(report,indent=2))
    print(report)
