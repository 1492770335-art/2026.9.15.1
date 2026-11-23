"""Rebuild price-impact and return-curve stages at their historical training cutoffs."""
import json
import numpy as np
import torch
import experiment_peer_matching_contrast_v1 as match
import experiment_pooled_price_impact_v1 as pooled
import experiment_source_alpha_impact_v1 as source
import experiment_term_structure_net_v1 as curve
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'walkforward_impact_stack_v1'
roll=match.roll
book=match.spread.book
MODELS=(pooled,source,curve)


def side_path(model,d):
    return (OUT/f'{model.OUT.name}_{d}.npy' if d<12 else model.OUT/f'side_{d}.npy')


def write_side(model,d,pred,short):
    np.testing.assert_allclose(pred[:len(short)],short,rtol=0,atol=4e-5)
    path=side_path(model,d)
    with path.with_suffix('.tmp').open('wb') as f:
        np.save(f,pred.astype('float32'))
    path.with_suffix('.tmp').replace(path)


def prepare():
    for mod in (pooled,source):
        if all(side_path(mod,d).exists() for d in range(8,12)):continue
        if mod is pooled:
            proj,vol=pooled.exposure(8)
            local,cross=pooled.train(8,proj,vol)
            def infer(raw):
                z=pooled.states(raw)
                return (z@local+(pooled.build(z,proj)@cross).reshape(-1,50,3))*vol[None,:,None]
        else:
            proj,vol,local,axes,coef=source.fit(8)
            def infer(raw):
                z=source.states(raw)
                loc=z@local
                return (loc+(source.design(z,loc,proj,axes)@coef).reshape(-1,50,3))*vol[None,:,None]
        for d in range(8,12):
            raw=book.rawday(d)
            write_side(mod,d,infer(raw),infer(raw[:213]))
        print('rebuilt cutoff 8',mod.OUT.name,flush=True)
    if not all(side_path(curve,d).exists() for d in range(8,12)):
        # Only the checkpoint output directory of the old trainer is redirected.
        directory=curve.OUT
        curve.OUT=OUT
        try:
            ck=OUT/'net_8.pth'
            if ck.exists():
                state=torch.load(ck,map_location='cpu',weights_only=True)
                net=curve.CurveNet(state['local.0.weight'].shape[1]).cuda().eval()
                net.load_state_dict(state)
            else:net=curve.train(8)
        finally:
            curve.OUT=directory
        for d in range(8,12):
            q=curve.infer(net,d)
            n=213
            raw=book.rawday(d)[:n]
            feat=curve.v1.features(raw)
            t=np.arange(n)
            x,v=curve.feature_rows(raw,feat,t)
            with torch.inference_mode():
                r,m=net(torch.from_numpy(x).cuda().float(),(.4+.6*torch.from_numpy(v).cuda()).clamp(.5,3))
                small=(r+net.beta[None,:,None]*m[:,None,:])[:,:,curve.J].cpu().numpy()*(np.sqrt(curve.H[curve.J])*curve.v1.SCALE)
            write_side(curve,d,q,small)
        print('rebuilt cutoff 8',curve.OUT.name,flush=True)


def blend(p,day,weights,t):
    for mod,w in zip(MODELS,weights):
        n=np.load(side_path(mod,day),mmap_mode='r')[t]
        p=pooled.v1.blend(p,n,w).astype('float32')
    return p


def weights(history):
    if not history:return np.zeros((len(MODELS),2,3))
    ps=[row[0] for row in history]
    out=[]
    for i in range(len(MODELS)):
        rows=[(p,row[1],row[2+i]) for p,row in zip(ps,history)]
        _,w=pooled.fits.coefficients(rows)
        # A bounded update at each stage preserves earlier neural and opening forecasts.
        w=.65*w
        ps=[pooled.v1.blend(p,n,w) for p,_,n in rows]
        out.append(w)
    return np.stack(out)


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2);prepare()
    history=[];mixing=[];blocks=[];daily=[]
    for end in (8,10,12,14,16):
        w=weights(history)
        c=roll.fit(mixing)
        rows=[]
        if end==16:
            np.savez(OUT/'frozen.npz',impact=w,reference=c)
        for d in range(end,end+(4 if end==16 else 2)):
            fs=roll.sources(d)
            t=np.arange(len(fs[0]))
            p=roll.predict(fs,t,c);q=blend(p,d,w,t)
            y=labels(d).reshape(q.shape)
            if d<16:
                tr=t[d%8::8]
                history.append((p[tr],y[tr],*[np.load(side_path(m,d),mmap_mode='r')[tr] for m in MODELS]))
                mixing.append(roll.moments(fs,y,t[d%5::5]))
            np.save(OUT/f'pred_{d}.npy',q)
            row=tuple(a.reshape(-1,3) for a in (p,q,y))
            daily.append(measure([row]));rows.append(row)
        blocks.append(measure(rows))
        print(end+1,'score',blocks[-1]['baseline_score'],blocks[-1]['score'],'y3',blocks[-1]['base'][2],blocks[-1]['candidate'][2],flush=True)
    report=dict(sources=[m.OUT.name for m in MODELS],fold_start=[9,11,13,15,17],
                score=blocks,daily=daily,formula='chronological neural + pooled impact + source impact + curve')
    (OUT/'results.json').write_text(json.dumps(report,indent=2))


if __name__=='__main__':main()
