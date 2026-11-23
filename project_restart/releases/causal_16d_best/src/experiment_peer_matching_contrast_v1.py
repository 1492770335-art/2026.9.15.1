"""Isolate the change from training on matched peer spreads against an identical uniform-peer network."""
import json
import numpy as np
import torch
import experiment_synthetic_peer_spreads_v1 as spread
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'peer_matching_contrast_v1'
roll=spread.roll


def rotated(weights):
    # Rotate stock identities only within unconditional volatility quintiles.
    sig=np.mean([spread.book.rawday(d)[:,:,0].std(0,dtype=float) for d in range(8)],axis=0)
    order=np.argsort(sig).reshape(5,10)
    perm=np.empty(50,int)
    perm[order.ravel()]=np.roll(order,3,axis=1).ravel()
    return weights[np.ix_(perm,perm)]


def apply(p,a,b):
    signal=.30*(b[:,:,2]-a[:,:,2])+.70*(b[:,:,3]-a[:,:,3])
    q=p.copy()
    q[:,:,2]+=.18*spread.innovation(p,signal[:,:,None])[:,:,0]
    return q


def load():
    parent=spread.audit.incumbent.load()
    ops=[spread.operators(m) for m in np.load(OUT/'portfolios.npy')]
    models=[]
    for i in range(2):
        state=torch.load(OUT/f'model_{i}.pth',map_location='cpu',weights_only=True)
        m=spread.SpreadNet(state['encoder.0.weight'].shape[1]).cuda().eval()
        m.load_state_dict(state);models.append(m)
    def predict(raw,p):
        q=parent(raw,p);t=np.arange(len(raw))
        a,b=[spread.infer(m,o,raw,t) for m,o in zip(models,ops)]
        return apply(q,a,b)
    return predict


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    peer=spread.portfolios()
    sets=[(np.ones((50,50))-np.eye(50))/49,peer,rotated(peer)]
    ops=[spread.operators(a) for a in sets]
    history=[];folds=[];days=[]
    for end,stop in ((8,12),(12,14),(14,16)):
        ds=spread.dataset(end)
        models=[spread.train(ds,o) for o in ops]
        del ds
        for start in range(end,stop,2):
            blocks=[[[],[]] for _ in range(2)]
            w=roll.fit(history)
            for d in (start,start+1):
                fs=roll.sources(d);raw=spread.book.rawday(d);t=np.arange(d%4,len(raw),4)
                y=labels(d).reshape(-1,50,3)
                base=roll.predict(fs,t,w)
                ref=roll.predict(fs,t,roll.prior())
                history.append(roll.moments(fs,y,np.arange(d%5,len(raw),5)))
                n=[spread.infer(m,o,raw,t) for m,o in zip(models,ops)]
                for k,p in enumerate((base,ref)):
                    for j,side in enumerate(n[1:]):
                        q=apply(p,n[0],side)
                        row=tuple(a.reshape(-1,3) for a in (p,q,y[t]))
                        blocks[k][j].append(row)
                        if k==j==0:
                            s=measure([row]);days.append(s['candidate'][2]-s['base'][2])
                            if d==start:
                                tt=t[t<167]
                                short=[spread.infer(m,o,raw[:167],tt) for m,o in zip(models[:2],ops[:2])]
                                np.testing.assert_allclose(q[:len(tt)],apply(p[:len(tt)],*short),rtol=0,atol=2e-6)
            folds.append([[measure(rs) for rs in pair] for pair in blocks])
            print(start+1,[[s['candidate'][2]-s['base'][2] for s in pair] for pair in folds[-1]],flush=True)
    g=np.array([[[s['candidate'][2]-s['base'][2] for s in pair] for pair in f] for f in folds])
    keep=bool((g[:,:,0].min()>0) and ((g[:,:,0]-g[:,:,1]).mean(0)>0).all() and
              ((g[:,0,0]>g[:,0,1]).sum()>=3) and (np.array(days)>0).sum()>=5)
    (OUT/'results.json').write_text(json.dumps(dict(benchmarks=['adaptive','fixed'],arms=['matched','permuted'],
                 retained=keep,folds=folds,daily=days),indent=2))
    print('retained',keep,flush=True)
    if keep:
        np.save(OUT/'portfolios.npy',np.stack(sets[:2]))
        ds=spread.dataset(16)
        for j,o in enumerate(ops[:2]):
            torch.save(spread.train(ds,o).cpu().state_dict(),OUT/f'model_{j}.pth')
        del ds
        replay=roll.gls.best.walk.replay
        res=replay(load())
        (OUT/'raw_replay.json').write_text(json.dumps(res,indent=2))
        old=json.loads((spread.audit.incumbent.OUT/'reload_verification.json').read_text())['pooled']['score']
        print('official',res['pooled']['score'],res['pooled']['score']-old,flush=True)
        if res['pooled']['score']>old:
            check=replay(load())
            np.testing.assert_allclose(res['pooled']['candidate'],check['pooled']['candidate'],rtol=0,atol=1e-10)
            (OUT/'reload_verification.json').write_text(json.dumps(check,indent=2))


if __name__=='__main__':
    main()
