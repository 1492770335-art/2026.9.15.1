"""Rebuild an inner-fold reference from cutoff-specific checkpoints and label-independent weights."""
import hashlib
import json
import numpy as np
import torch
import experiment_replicated_expert_consensus_v1 as experts
import experiment_state_conditional_errors_v1 as state
import experiment_curve_market_components_v1 as historical
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'cutoff_reference_panel_v1'
walk=state.walk
CLASSES=(experts.phase.BookNet,experts.sym.PressureNet,experts.spread.PairNet,experts.bar.ScaleNet)
KEYS=('current.0.weight','gate.0.weight','state.0.weight',None)


def cutoff(day):
    if not 8<=day<16:raise ValueError('reference is only defined for inner validation dates')
    return 8 if day<12 else (12 if day<14 else 14)


def models(end):
    result=[]
    for i,(name,cls,key) in enumerate(zip(experts.experts.SOURCES,CLASSES,KEYS)):
        path=ART/name/f'net_{end}.pth'
        st=torch.load(path,map_location='cpu',weights_only=True)
        model=(cls(st[key].shape[1]) if key else cls()).cuda().eval()
        model.load_state_dict(st)
        if i==1:
            fn=lambda raw,net=model:experts.phase.infer(net,raw,make_features=experts.sym.features)
        elif i==3:
            fn=lambda raw,net=model:experts.bar.predict(net,raw)
        else:
            fn=lambda raw,net=model:experts.phase.infer(net,raw)
        result.append((fn,hashlib.sha256(path.read_bytes()).hexdigest()))
    return result


def build():
    OUT.mkdir(exist_ok=True)
    audit=[]
    for end,stop in ((8,12),(12,14),(14,16)):
        nets=models(end)
        for d in range(end,stop):
            raw=walk.book.rawday(d)
            all_n=[]
            errors=[]
            for (fn,sha),src in zip(nets,experts.experts.SOURCES):
                q=fn(raw)
                n=173
                # Some model implementations need four-way split inputs longer than a few ticks.
                err=float(np.max(abs(q[:n]-fn(raw[:n]))))
                if not np.isfinite(q).all() or err>1e-4:raise ValueError((src,d,err))
                all_n.append(q)
                old=np.load(ART/src/f'side_{d}.npy',mmap_mode='r')
                errors.append(dict(source=src,sha256=sha,prefix_error=err,cached_difference=float(np.max(abs(old-q)))))
            y=labels(d).reshape(-1,50,3)
            opening=np.load(ART/'opening_oof_cache_v1'/f'day_{d:02d}.npy').reshape(y.shape)
            relative=sum(v-v.mean(1,keepdims=True) for v in all_n)/4
            shared=sum(all_n[i].mean(1,keepdims=True) for i in (0,1,3))/3
            neural=(relative+shared).astype('float32')
            strong=(.65*opening+.35*neural).astype('float32')
            np.save(OUT/f'panel_{d}.npy',strong)
            np.save(OUT/f'neural_{d}.npy',neural)
            legacy=walk.blend.panel.legacy_early(d)
            row=lambda p:tuple(v.reshape(-1,3) for v in (legacy,p,y))
            sc=measure([row(strong)])
            audit.append(dict(day=d+1,network_training_end=end,opening_training_end=8 if d<12 else 12,
                              experts=errors,legacy=sc['baseline_score'],panel=sc['score'],opening=measure([row(opening)])['score'],
                              neural=measure([row(neural)])['score']))
            print(d+1,sc['baseline_score'],sc['score'],flush=True)
    return audit


def gate_effect():
    saved=np.load(historical.OUT/'weights.npy')
    hist=json.loads((historical.OUT/'results.json').read_text())
    result=[]
    for d in range(12,16):
        p,z=historical.delta(d)
        old=p+np.einsum('tkj,jk->tk',z,saved)[:,None]
        past=np.zeros_like(saved) if d<14 else (np.array(hist['14'])>0).astype(float)
        alt=p+np.einsum('tkj,jk->tk',z,past)[:,None]
        y=labels(d).reshape(p.shape)
        result.append(dict(day=d+1,max_change=float(np.max(abs(old-alt))),score=measure([tuple(v.reshape(-1,3) for v in (old,alt,y))])))
    return dict(stored_mask=saved.tolist(),isolated_comparison=result)


if __name__=='__main__':
    audits=build()
    report=dict(audit=audits,legacy_gate=gate_effect(),panel_formula='0.65 chronological opening + 0.35 cutoff neural panel')
    (OUT/'audit.json').write_text(json.dumps(report,indent=2))
