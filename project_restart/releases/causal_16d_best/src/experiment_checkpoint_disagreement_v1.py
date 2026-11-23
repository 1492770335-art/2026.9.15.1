"""Evaluate changes to neural expert predictions caused by including more recent training dates."""
import json
import numpy as np
import torch
import experiment_delayed_tail_band_skill_v1 as early
import experiment_eight_date_expert_contrasts_v1 as readout
import experiment_phase_future_book_v1 as phase
import experiment_pairwise_spread_decoder_v1 as pair
import experiment_pairwise_book_tree_v1 as tree
import experiment_repricing_adjusted_book_flow_v1 as flow
import replay_verified_forecast_parent_v1 as replay
from experiment_causal_cross_horizon_ridge_v1 import ART, corr, labels, measure

OUT = ART / 'checkpoint_disagreement_v1'
PARENT = ART / 'directional_depth_tree_v1'


def networks(end):
    result=[]
    device='cuda' if torch.cuda.is_available() else 'cpu'
    for directory,cls,key in ((phase.OUT,phase.BookNet,'current.0.weight'),(pair.OUT,pair.PairNet,'state.0.weight')):
        family=[]
        for cutoff in (8,12,14,16):
            if cutoff>end:
                break
            state=torch.load(directory/f'net_{cutoff}.pth',map_location='cpu',weights_only=True)
            net=cls(state[key].shape[1])
            net.load_state_dict(state)
            family.append((cutoff,net.to(device).eval()))
        result.append(family)
    return result


@torch.inference_mode()
def features(raw,p,end,models):
    ticks=np.arange(len(raw))
    dev=next(models[0][0][1].parameters()).device
    # All expert checkpoints receive identical causal features.
    old=[np.empty_like(p) for family in models]
    recent=[np.empty_like(p) for family in models]
    for offset in range(4):
        ids=ticks[offset::4]
        fx,risk=phase.spline.features(raw,ids)
        for i in range(0,len(ids),32):
            t=ids[i:i+32]
            x=torch.from_numpy(fx[i:i+32]).to(dev).float()
            vol=torch.from_numpy((.45+.55*risk[i:i+32]).clip(.55,3)).to(dev)
            gate=torch.from_numpy(np.exp(-t[:,None]/400).astype('float32')).to(dev)
            for j,family in enumerate(models):
                weights=np.array([2**(-(end-cutoff)/4) for cutoff,net in family[:-1]])
                prev=0
                for w,(_,net) in zip(weights/weights.sum(),family[:-1]):
                    prev=prev+float(w)*net(x,vol,gate)[0]
                old[j][t]=prev.cpu().numpy()
                recent[j][t]=family[-1][1](x,vol,gate)[0].cpu().numpy()
    mk=[];rel=[]
    for j,(u,v) in enumerate(zip(old,recent)):
        change=u-v
        a=change.mean(1,keepdims=True)
        b=tree.innovation(p,change-a)
        agree=np.tanh(u*v/(.035+.4*(u*u+v*v)))
        # Only the phase expert has a market head.
        if j==0:
            mk.extend((a[:,0],a[:,0]*agree.mean(1)))
        mod=b*agree
        rel.extend((b,mod-mod.mean(1,keepdims=True)))
    return np.concatenate(mk,axis=1),np.concatenate(rel,axis=2)


def main():
    OUT.mkdir(exist_ok=True)
    history=[];report={}
    for end,stop in ((12,14),(14,16),(16,20)):
        nets=networks(end)
        if end>=14:
            w=flow.fit(history)
        if end==16:
            mask=np.min(gains,axis=0)>0
            report['eligible']=mask.tolist()
            if not mask.any():
                (OUT/'results.json').write_text(json.dumps(report,indent=2))
                return
            w=[a*gate for a,gate in zip(w,mask)]
            np.savez(OUT/'readout.npz',market=w[0],relative=w[1])
        gains=[];rows=[]
        def advance(raw,p):
            a,b=features(raw,p,end,nets)
            return (p+(a@w[0])[:,None]+b@w[1]).astype('float32')
        for d in range(end,stop):
            p=np.load(PARENT/f'pred_{d}.npy') if end==16 else early.base(d)
            raw=tree.rawday(d)
            a,b=features(raw,p,end,nets)
            if d==end:
                check=features(raw[:137],p[:137],end,nets)
                for full,head in zip((a,b),check):
                    np.testing.assert_allclose(full[:137],head,atol=1e-5)
            if end>=14:
                terms=((a@w[0])[:,None],b@w[1])
                q=(p+sum(terms)).astype('float32')
                if end==16:
                    np.save(OUT/f'pred_{d}.npy',q)
            y=labels(d).reshape(p.shape)
            if end<16:
                history.append(readout.moments(p,y,(a,b)))
            if end>=14:
                row=tuple(v.reshape(-1,3) for v in (p,q,y))
                rows.append(row)
                if end==14:
                    base=corr(row[0],row[2])
                    gains.append([corr((p+c).reshape(-1,3),row[2])-base for c in terms])
        if end>=14:
            report[str(stop)]=measure(rows)
            print(stop,report[str(stop)],flush=True)
            (OUT/'results.json').write_text(json.dumps(report,indent=2))
    if report['20']['gain']>0:
        replay.main(PARENT,OUT,advance)


if __name__=='__main__':
    main()
