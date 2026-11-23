"""Compare causal, intraday expert routing with the same four equally weighted tail forecasts."""
import json
import numpy as np
import experiment_pairwise_book_tree_v1 as book
import experiment_tail_portfolio_filter_v1 as panel
from experiment_causal_cross_horizon_ridge_v1 import ART,labels
from experiment_predictive_book_modes_v1 import validation
from experiment_eight_date_expert_contrasts_v1 import SOURCES

OUT=ART/'matured_expert_tail_v1'


def directions(raw,p,forecasts):
    tail=np.stack([v[:,:,2]-panel.RATIO*v[:,:,1] for v in forecasts],axis=-1)
    tail-=tail.mean(1,keepdims=True)
    average=tail.mean(2)
    delta=tail-average[:,:,None]
    x=p[:,:,2]-panel.RATIO*p[:,:,1]
    fixed=average-(x-x.mean(1,keepdims=True))
    ret=raw[:,:,0].astype(float)-.00136
    path=ret.cumsum(axis=0)
    n=len(raw);tick=np.arange(n);stock=np.arange(50)
    score=np.zeros_like(delta);keep=np.zeros_like(x)
    if n>60:
        actual=.161*(path[60:]-path[20:-40])
        actual-=actual.mean(1,keepdims=True)
        error=(actual-average[:-60]).clip(-3,3)
        reward=delta[:-60]*(2*error[:,:,None]-delta[:-60])
        # Each stock contributes nonoverlapping intervals, staggered across stocks.
        keep[60:]=(tick[:-60,None]+stock)%40==0
        score[60:]=reward*keep[60:,:,None]
    def cs(a):
        return np.concatenate((np.zeros_like(a[:1],dtype=float),a.cumsum(0,dtype=float)))
    total=cs(score);square=cs(score**2);count=cs(keep)
    directions=[fixed]
    for width in (600,2400):
        end=tick+1;start=(end-width).clip(0)
        v=total[end]-total[start];v2=square[end]-square[start];k=count[end]-count[start]
        broad=v.sum(1,keepdims=True)
        # Square-root scores are shrunk more for the shared component, where assets co-move.
        group=(broad/np.sqrt(12*v2.sum(1,keepdims=True)+.45)).clip(-2,2)
        own=(v/np.sqrt(v2+.08)).clip(-2,2)*k[:,:,None]/(k[:,:,None]+12)
        weight=np.exp((group+own)*.65)
        chosen=(delta*(weight/weight.sum(2,keepdims=True))).sum(2)
        directions.append(chosen-chosen.mean(1,keepdims=True))
    return np.stack(directions,axis=-1).astype('float32')


def fit(history):
    g,b=(np.stack(v) for v in zip(*history))
    def solve(grams,rhs):
        reg=.75*np.diag(grams.mean(0).diagonal().clip(.00001,None))
        per=np.linalg.solve(grams+reg,rhs[:,:,None])[:,:,0]
        pooled=np.linalg.solve(grams.mean(0)+reg,rhs.mean(0))
        return (.65*(.65*pooled+.35*np.median(per,axis=0))*abs(np.sign(per).mean(0))).clip(0,.6)
    fixed=solve(g[:,:1,:1],b[:,:1])
    # Use exactly the same equal-weight component in both branches.
    extra=solve(g[:,1:,1:],b[:,1:]-g[:,1:,0]*fixed)
    return np.r_[fixed,extra].astype('float32')


def added(p,z,w):
    q=p.copy();q[:,:,2]+=z@w
    return q


def fixed_forecast(p,forecasts,weight):
    tail=np.stack([v[:,:,2]-panel.RATIO*v[:,:,1] for v in forecasts],axis=-1)
    tail-=tail.mean(1,keepdims=True)
    original=p[:,:,2]-panel.RATIO*p[:,:,1]
    q=p.copy()
    q[:,:,2]+=weight*(tail.mean(2)-original+original.mean(1,keepdims=True))
    return q


def main(fixed=False):
    output=ART/'equal_expert_relative_tail_v1' if fixed else OUT
    output.mkdir(exist_ok=True);history=[];report={'folds':[]};daily=[]
    for end in (8,10,12,14):
        w=fit(history) if history else None;rows=[]
        for d in range(end,end+2):
            raw=book.rawday(d);p=panel.early(d);y=labels(d).reshape(p.shape)
            sources=[np.load(ART/name/f'side_{d}.npy') for name in SOURCES]
            z=directions(raw,p,sources)
            if d==end:
                head=directions(raw[:297],p[:297],[a[:297] for a in sources])
                np.testing.assert_allclose(z[:297],head,rtol=0,atol=2e-6)
            if w is not None:
                plain=added(p,z[:,:,:1],w[:1]);dynamic=added(p,z,w)
                daily.append(panel.scores(plain if fixed else dynamic,y,0)-panel.scores(p,y,0))
                rows.append((p,plain,dynamic,y))
            design=z[::3].reshape(-1,3).astype(float)
            e=panel.target_residual(p[::3],y[::3]).ravel()
            history.append((design.T@design/len(e),design.T@e/len(e)))
        if rows:
            p,plain,dynamic,y=map(np.concatenate,zip(*rows))
            fold=dict(days=[end+1,end+2],baseline=panel.scores(p,y,0).tolist(),
                      equal=panel.scores(plain,y,0).tolist(),routed=panel.scores(dynamic,y,0).tolist())
            report['folds'].append(fold);print(fold,flush=True)
    gains=np.array([np.subtract(f['routed'],f['baseline']) for f in report['folds']])
    routing=np.array([np.subtract(f['routed'],f['equal']) for f in report['folds']])
    gate=bool((gains>0).all() and (routing>0).sum(0).min()>=2 and (routing.mean(0)>0).all()
              and ((np.array(daily)>0).sum(0)>=4).all())
    if fixed:
        equal=np.stack([np.subtract(f['equal'],f['baseline']) for f in report['folds']])
        gate=bool((equal>0).all() and ((np.array(daily)>0).sum(0)>=4).all())
    report['retained']=gate;path=output/'results.json';path.write_text(json.dumps(report,indent=2))
    if gate:
        import experiment_replicated_expert_consensus_v1 as consensus
        models=consensus.sources();w=fit(history)
        if fixed:w[1:]=0
        np.save(output/'weights.npy',w)
        report['weights']=w.tolist()
        path.write_text(json.dumps(report,indent=2))
        def advance(raw,p):
            forecasts=[fn(raw) for fn in models]
            if fixed:return fixed_forecast(p,forecasts,w[0])
            return added(p,directions(raw,p,forecasts),w)
        report['development']=validation(advance)
        path.write_text(json.dumps(report,indent=2));print(report['development'],flush=True)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--fixed',action='store_true')
    main(parser.parse_args().fixed)
