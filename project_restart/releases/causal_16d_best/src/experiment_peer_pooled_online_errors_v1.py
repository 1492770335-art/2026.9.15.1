"""Borrow completed far-return errors from matched stocks after reconstructing the stock calibration stage."""
import json
import numpy as np
import experiment_walkforward_impact_stack_v1 as core
import experiment_intraday_relative_calibration_v1 as online
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, measure

OUT=ART/'peer_pooled_online_errors_v1'
match=core.match
incumbent=match.spread.audit.incumbent
BLOCK=12
RATIO=.160943/.272164


def legacy_stats(p,y,x):
    e=y.astype(float)-p;e-=e.mean(1,keepdims=True)
    return (np.einsum('tski,tskj->kij',x,x,dtype=float)/len(p)/50,
            np.einsum('tski,tsk->ki',x,e,dtype=float)/len(p)/50)


def legacy_fit(hist):
    if not hist:return np.zeros((3,4))
    grams,bs=map(lambda z:np.mean(z,axis=0),zip(*hist))
    return np.stack([np.linalg.solve(g+.9*np.diag(g.diagonal().clip(1e-6)),b)
                     for g,b in zip(grams,bs)]).clip(-1.,1.)


def borrowed(raw,p,graphs):
    feat,vol,ret=online.context(raw,p)
    near=RATIO*p[:,:,1];far=p[:,:,2]-near
    t=np.arange(len(p))
    z=np.concatenate((feat[:,:,[0,1,6,7,8,9,10,13,14]],np.stack((
       (near-near.mean(1,keepdims=True))/vol,(far-far.mean(1,keepdims=True))/vol),axis=2)*3),axis=2).clip(-3,3)
    # The 21..60 interval has just finished for the forecast issued at t-60.
    c=ret.cumsum(0);origin=(t-60).clip(0)
    a=z[origin].copy();a[:60]=0
    realized=.160943*(c-c[np.minimum(origin+20,t)])
    realized-=realized.mean(1,keepdims=True)
    residual=((realized-(far-far.mean(1,keepdims=True))[origin])/vol[origin]).clip(-2,2)
    blocks=(len(p)+BLOCK-1)//BLOCK
    x=np.zeros((blocks*BLOCK,50,z.shape[2]));x[:len(p)]=a
    e=np.zeros((blocks*BLOCK,50));e[:len(p)]=residual
    x=x.reshape(blocks,BLOCK,50,-1)
    gram=np.einsum('btsi,btsj->bsij',x,x,optimize=True)
    rhs=np.einsum('btsi,bts->bsi',x,e.reshape(blocks,BLOCK,50),optimize=True)
    results=[[] for _ in graphs]
    for win,shrink in ((35,330),(125,700)):
        g=online.rolling(gram,win);b=online.rolling(rhs,win)
        uniform=g.mean(1);rb=b.mean(1)
        prior=np.linalg.solve(uniform+shrink*np.eye(z.shape[2]),rb[:,:,None])[:,:,0]
        # Discount the part explained by the market-wide calibration slope.
        innovation=b-(g@prior[:,None,:,None])[:,:,:,0]
        for parts,P in zip(results,graphs):
            # Compare two priors on the same stock's personal slope. Each graph
            # has zero self-weight and each row sums to one.
            def estimate(peers,rg):
                h=.4*g+.6*peers
                return np.linalg.solve(h+shrink*np.eye(z.shape[2]),(.4*innovation+.6*rg)[:,:,:,None])[:,:,:,0]
            def pool(u):
                return (P@u.reshape(blocks,50,-1)).reshape(u.shape)
            matched=estimate(pool(g),pool(innovation))
            other=estimate((50*uniform[:,None]-g)/49,(innovation.sum(1,keepdims=True)-innovation)/49)
            delta=((matched-other)[t//BLOCK]*z).sum(2)*vol
            parts.append((delta-delta.mean(1,keepdims=True)).astype('float32'))
    return [np.stack(a,axis=2) for a in results]


def record(p,y,z):
    b=p[:,:,2].astype(float);v=y[:,:,2].astype(float)
    b-=b.mean();v-=v.mean()
    slope=((b*v).mean()/(b*b).mean()).clip(.35,3)
    r=v/slope-b
    x=z.reshape(-1,2).astype(float)
    return x.T@x/len(x),x.T@(r-r.mean(1,keepdims=True)).ravel()/len(x)


def weights(history):
    gs,bs=map(np.stack,zip(*history));g=gs.mean(0)
    reg=.9*np.diag(g.diagonal().clip(.00002))
    daily=np.linalg.solve(gs+reg,bs[:,:,None])[:,:,0]
    return (.65*(.6*np.linalg.solve(g+reg,bs.mean(0))+.4*np.median(daily,0))*abs(np.sign(daily).mean(0))).clip(-.8,.8)


def load():
    parent=incumbent.load()
    with np.load(OUT/'model.npz') as st:
        graph,w=st['graph'],st['weight']
    def apply(raw,p):
        base=parent(raw,p);q=base.copy()
        q[:,:,2]+=borrowed(raw,base,[graph])[0]@w
        return q
    return apply


def main():
    OUT.mkdir(exist_ok=True)
    graph=match.spread.portfolios().astype(float)
    graphs=[graph,match.rotated(graph)]
    h0=[];hist=[[],[]];calibrated=[];folds=[];days=[]
    for end in (8,10,12,14,16):
        c=legacy_fit(h0)
        ws=[weights(h) for h in hist] if hist[0] and end<16 else []
        rows=[];blocks=[[],[]]
        for d in range(end,end+(4 if end==16 else 2)):
            raw=core.book.rawday(d)
            p=np.load(core.OUT/f'pred_{d}.npy')
            # Only this stage's coefficients fitted on earlier dates are used.
            x=online.adaptive(raw,p.astype(float))
            base=(p+np.einsum('tskj,kj->tsk',x,c)).astype('float32')
            np.save(OUT/f'reference_{d}.npy',base)
            y=labels(d).reshape(base.shape)
            rows.append(tuple(a.reshape(-1,3) for a in (p,base,y)))
            if d>=16:continue
            h0.append(legacy_stats(p,y,x))
            z=borrowed(raw,base,graphs)
            if d==end:
                cut=221
                xx=online.adaptive(raw[:cut],p[:cut].astype(float))
                short=(p[:cut]+np.einsum('tskj,kj->tsk',xx,c)).astype('float32')
                np.testing.assert_allclose(short,base[:cut],rtol=0,atol=2e-6)
                early=borrowed(raw[:cut],short,graphs)
                for full,head in zip(z,early):np.testing.assert_allclose(full[:cut],head,rtol=0,atol=2e-6)
            for j,feat in enumerate(z):
                hist[j].append(record(base,y,feat))
                if not ws:continue
                q=base.copy();q[:,:,2]+=feat@ws[j]
                row=tuple(a.reshape(-1,3) for a in (base,q,y))
                blocks[j].append(row)
                if j==0:
                    s=measure([row]);days.append(s['candidate'][2]-s['base'][2])
        calibrated.append(measure(rows))
        if ws:
            folds.append([measure(b) for b in blocks])
        print(end+1,'reference',calibrated[-1]['score'],'y3',calibrated[-1]['candidate'][2],
              'peer gains',[m['candidate'][2]-m['base'][2] for m in folds[-1]] if ws else [],flush=True)
    g=np.array([[s['candidate'][2]-s['base'][2] for s in f] for f in folds])
    keep=bool((g[:,0].min()>0) and (g[:,0]-g[:,1]).mean()>0 and (np.array(days)>0).sum()>=4)
    report=dict(reference=calibrated,folds=folds,retained=keep,arms=['matched','permuted'],daily_gains=days)
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    print('retained',keep,flush=True)
    if keep:
        np.savez(OUT/'model.npz',graph=graph,weight=weights(hist[0]))
        replay=core.roll.gls.best.walk.replay
        result=replay(load())
        (OUT/'raw_replay.json').write_text(json.dumps(result,indent=2))
        prev=json.loads((incumbent.OUT/'reload_verification.json').read_text())['pooled']['score']
        print('official',result['pooled']['score'],result['pooled']['score']-prev,flush=True)
        if result['pooled']['score']>prev:
            check=replay(load())
            np.testing.assert_allclose(check['pooled']['candidate'],result['pooled']['candidate'],rtol=0,atol=1e-10)
            (OUT/'reload_verification.json').write_text(json.dumps(check,indent=2))


if __name__=='__main__':main()
