"""Measure rolling Y3 gains of the two corrections in current high-activity and ordinary states."""
import json
import numpy as np
import torch

import experiment_y3_transition_implied_moves_v1 as event
import y3_clock_and_depth_references as final

trial=final.trial
mid=final.previous
base=mid.previous
OUT=final.incumbent.OUT


def report(g):
    c=g[1:,1:]-np.outer(g[0,1:],g[0,1:])/g[0,0]
    # Columns: original Y3, mass update in low/high activity, level update in low/high, official Y3.
    axis=np.eye(5)
    original=axis[0]
    effects=[axis[1],axis[2],axis[3],axis[4]]
    def ic(w):
        return float(w@c[:-1,-1]/np.sqrt((w@c[:-1,:-1]@w)*c[-1,-1]))
    def corr(a,b):
        q=c[:-1,:-1]
        return float(a@q@b/np.sqrt((a@q@a)*(b@q@b)))
    return dict(base=ic(original),weighted_by_row_gains=[ic(original+a)-ic(original) for a in effects],
                mass_gain=ic(original+effects[0]+effects[1])-ic(original),
                depth_after_mass=ic(original+sum(effects))-ic(original+effects[0]+effects[1]),
                total_gain=ic(original+sum(effects))-ic(original),update_correlation=corr(effects[0]+effects[1],effects[2]+effects[3]))


def run():
    # Read only the transition decoder weights learned on days 1–4 for the activity mask.
    with np.load(mid.incumbent.OUT/'model.npz') as f:
        st={k:f[k] for k in f if k.startswith(('n0_','n1_')) or k in ('k','scale')}
    nets=event.restore(st)
    grams=np.zeros((2,7,7))
    days=[]
    daily_grams=[]
    counts=np.zeros(2)
    for day in range(8,16):
        raw=trial.clock.book.rawday(day)
        t=np.arange(day%4,len(raw)-60,4)
        _,s=event.tape(raw,st,nets)
        high=s[t,:,1]>=.5
        p,q,r=[module.references(day,raw) for module in (base,mid,final)]
        y=trial.clock.labels(day).reshape(-1,50,3)[t,:,2]
        daily=[]
        current=[]
        for i in range(2):
            a,b,c=[v[i][t,:,2].astype(float) for v in (p,q,r)]
            h=(b-a)
            k=(c-b)
            z=np.stack((np.ones_like(a),a,h*(~high),h*high,k*(~high),k*high,y),2).reshape(-1,7)
            g=z.T@z
            current.append(g)
            grams[i]+=g
            daily.append(report(g))
        days.append(daily)
        daily_grams.append(current)
        counts+=np.array([high.size,high.sum()])
    summary=dict(days=list(range(9,17)),rows=int(counts[0]),high_fraction=counts[1]/counts[0],
                 effects=['activity correction on ordinary states','activity correction on high states',
                          'price-level correction on ordinary states','price-level correction on high states'],
                 pooled=[report(g) for g in grams],daily=days)
    # Fit independent rescaling on days 9–12 and apply those fixed coefficients to days 13–16.
    # The published model always keeps its original coefficients throughout this diagnostic.
    transfer=[]
    for j in range(2):
        a,b=[sum(pair[j] for pair in part) for part in (daily_grams[:4],daily_grams[4:])]
        def cov(g):
            return g[1:,1:]-np.outer(g[0,1:],g[0,1:])/g[0,0]
        earlier,later=cov(a),cov(b)
        baseline=np.ones(5)
        def ic(c,w):
            return float(w@c[:-1,-1]/np.sqrt((w@c[:-1,:-1]@w)*c[-1,-1]))
        def fit_effects(cols):
            c=earlier[:-1,:-1]
            beta=(baseline@earlier[:-1,-1])/(baseline@c@baseline)
            rhs=(earlier[:-1,-1]/beta-c@baseline)[cols]
            g=c[np.ix_(cols,cols)]
            change=np.linalg.solve(g+.75*np.diag(g.diagonal().clip(1e-9)),rhs)
            new=baseline.copy()
            new[cols]+=change.clip(-.5,.5)
            return dict(weights=new[1:].tolist(),fit_gain=ic(earlier,new)-ic(earlier,baseline),
                        forward_gain=ic(later,new)-ic(later,baseline))
        transfer.append({key:fit_effects(cols) for key,cols in
                         [('high_only',[2,4]),('ordinary_only',[1,3]),('both',[1,2,3,4])]})
    summary['scale_fit_days']=[9,12]
    summary['scale_evaluation_days']=[13,16]
    summary['scale_transfer']=transfer
    (OUT/'rolling_state_attribution.json').write_text(json.dumps(summary,indent=2))
    print('rolling state attribution',summary['high_fraction'],summary['pooled'],flush=True)


if __name__=='__main__':
    torch.set_num_threads(2)
    run()
