"""Use training dates' cross-stock error covariance to fit a shared residual forecast by generalized least squares."""
import json
import numpy as np
import experiment_personalized_exposure_transfer_v1 as incumbent
import experiment_architecture_family_portfolio_v1 as family
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'crossfit_error_precision_v1'
book,best=incumbent.book,incumbent.best


def design(s,v,p,t):
    a=np.tanh(s[t][:,:,[0,2,3,7,8,10,12,21,23,28,29,30,31,36,44,48,57,61]])
    avg=a.mean(1,keepdims=True)
    q=p[t];r=q-q.mean(1,keepdims=True)
    old=p[(t-18).clip(0)];change=r-(old-old.mean(1,keepdims=True))
    b=np.concatenate((a-avg,np.tanh(a*2)-np.tanh(avg*2),r,np.tanh(4*r),
                      np.tanh(4*change),r*np.log(v[t,:,None]),
                      r*np.tanh(4*q.mean(1,keepdims=True)),r*np.exp(-t[:,None,None]/300)),axis=2)
    return (b-b.mean(1,keepdims=True)).astype('float32')


def record(s,v,p,y,t):
    x=design(s,v,p,t);q=p[t].astype(float);b=y[t].astype(float)
    a=q-q.mean((0,1));target=b-b.mean((0,1))
    slope=((a*target).mean((0,1))/(a*a).mean((0,1))).clip(.4,2.5)
    residual=b/slope-q;residual-=residual.mean(1,keepdims=True)
    cov=np.stack([e.T@e/len(e) for e in np.moveaxis(residual,2,0)])
    return x,residual.astype('float32'),cov


def precision(cov,full):
    # Match each stock's diagonal weight in both arms. Only cross-stock terms differ.
    sd=np.sqrt(cov.diagonal().clip(.003))
    c=cov/np.outer(sd,sd)
    base=(1/sd).clip(.38,2.8);base/=base.mean()
    matrix=np.linalg.inv(.55*c+.45*np.eye(len(c))) if full else np.eye(len(c))
    norm=np.sqrt(matrix.diagonal())
    return (matrix*np.outer(np.sqrt(base)/norm,np.sqrt(base)/norm)).astype('float32')


def fit(rows,full):
    covs=np.stack([c for _,_,c in rows])
    result=[]
    for k in range(3):
        grams=[];rhs=[]
        for d,(x,e,_) in enumerate(rows):
            # Estimate this date's loss geometry from other completed training dates.
            cov=np.mean(np.delete(covs[:,k],d,axis=0),axis=0)
            matrix=precision(cov,full)
            ax=(matrix@x).reshape(-1,x.shape[2]);flat=x.reshape(ax.shape)
            grams.append(flat.T@ax/len(ax))
            rhs.append(ax.T@e[:,:,k].ravel()/len(ax))
        gs=np.stack(grams).astype(float);bs=np.stack(rhs).astype(float)
        g=gs.mean(0);reg=1.6*np.diag(g.diagonal().clip(.00005))
        pooled=np.linalg.solve(g+reg,bs.mean(0))
        by_day=np.linalg.solve(gs+reg,bs[:,:,None])[:,:,0]
        w=.5*(.7*pooled+.3*np.median(by_day,0))*abs(np.sign(by_day).mean(0))
        result.append(w.clip(-.2,.2))
    return np.column_stack(result).astype('float32')


def adjust(p,x,w):
    return (p+x@w).astype('float32')


def load():
    parent=incumbent.load();w=np.load(OUT/'weights.npy')
    def apply(raw,p):
        base=parent(raw,p);q=base.copy();s,v=book.state(raw)
        for k in range(min(len(raw),4)):
            t=np.arange(k,len(raw),4)
            q[t]=adjust(base[t],design(s,v,base,t),w)
        return q
    return apply


def main():
    OUT.mkdir(exist_ok=True)
    rows=[];folds=[];daily=[]
    for end in (8,10,12,14):
        ws=[fit(rows,full) for full in (False,True)] if rows else []
        block=[[[],[]] for _ in range(3)]
        for d in (end,end+1):
            raw=book.rawday(d);s,v=book.state(raw);t=np.arange(d%4,len(raw),4)
            y=labels(d).reshape(-1,50,3)
            panel=np.load(best.refs.OUT/f'panel_{d}.npy',mmap_mode='r')
            neural=np.load(best.refs.OUT/f'neural_{d}.npy',mmap_mode='r')
            event=np.load(family.side_path(family.event,d),mmap_mode='r')
            rt=np.arange(d%7,len(raw)-60,7);rows.append(record(s,v,panel,y,rt))
            for j,p in enumerate((panel,neural,.55*panel+.45*event)):
                x=design(s,v,p,t)
                for k,w in enumerate(ws):
                    q=adjust(p[t],x,w)
                    if k and not j and d==end:
                        ix=t<173;ss,vv=book.state(raw[:173])
                        part=adjust(p[t[ix]],design(ss,vv,p[:173],t[ix]),w)
                        np.testing.assert_allclose(q[ix],part,rtol=0,atol=2e-6)
                    row=tuple(a.reshape(-1,3) for a in (p[t],q,y[t]))
                    block[j][k].append(row)
                    if k and not j:
                        sc=measure([row]);daily.append(np.subtract(sc['candidate'],sc['base']))
        if ws:folds.append([[measure(a) for a in pair] for pair in block])
    g=np.array([[[np.subtract(s['candidate'],s['base']) for s in pair] for pair in f] for f in folds])
    keep=(g[:,0,1].min(0)>0)&(g[:,:,1].mean(0).min(0)>0)&((g[:,0,1]-g[:,0,0]).mean(0)>0)&((np.stack(daily)>0).sum(0)>=4)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,retained=keep.tolist()),indent=2))
    print('gains',g[:,0,1],'full minus diagonal',g[:,0,1]-g[:,0,0],'kept',keep,flush=True)
    if keep.any():
        np.save(OUT/'weights.npy',fit(rows,True)*keep)
        res=best.walk.replay(load())
        (OUT/'raw_replay.json').write_text(json.dumps(res,indent=2))
        old=json.loads((incumbent.OUT/'reload_verification.json').read_text())
        print('official',res['pooled'],res['pooled']['score']-old['pooled']['score'],flush=True)
        if res['pooled']['score']>old['pooled']['score']:
            rep=best.walk.replay(load())
            np.testing.assert_allclose(rep['pooled']['candidate'],res['pooled']['candidate'],rtol=0,atol=1e-10)
            (OUT/'reload_verification.json').write_text(json.dumps(rep,indent=2))


if __name__=='__main__':
    main()
