"""Replace the probability revision by balancing independently estimated memberships at inference."""
import json
import sys

import numpy as np
import torch

import y3_rank_payout_references as rank_history

rank = rank_history.incumbent
history,parent,base = rank.history,rank.parent,rank.base
OUT = parent.OUT.parent/'y3_rank_capacity_readout_v1'


def load_models():
    with np.load(rank.OUT/'model.npz') as f:
        assert rank.ARMS[int(f['pick'])+1]=='balanced probability contribution'
        nets = [rank.Model(int(f['width']),mode).cuda().eval() for mode in (1,2)]
        for i,net in enumerate(nets,1):
            net.load_state_dict({k:torch.from_numpy(f[str(i)+'_'+k]) for k in net.state_dict()})
        return nets,float(f['w'])


@torch.inference_mode()
def change(raw,nets,weight):
    x,ph = rank.inputs.features(raw,np.arange(len(raw)))
    output = []
    for i in range(0,len(x),96):
        v = torch.from_numpy(x[i:i+96]).cuda()
        s = torch.from_numpy(ph[i:i+96]).cuda()
        _,free,free_pay = nets[0](v,s)
        alt_pay = nets[1](v,s)[2]
        prob = free[:,:,2,:]
        balanced = prob
        for _ in range(4):
            balanced = balanced*5/balanced.sum(1,keepdim=True).clamp_min(1e-9)
            balanced = balanced/balanced.sum(-1,keepdim=True).clamp_min(1e-9)
        pay = .5*(free_pay[:,2,:]+alt_pay[:,2,:])
        output.append((((balanced-prob)*pay[:,None]).sum(-1)*weight).cpu().numpy())
    return np.concatenate(output)


def train():
    # Reuse the earlier cross-fitted coefficient without fitting a coefficient to the four evaluation days.
    import audit_y3_rank_capacity_effect_v1 as audit
    OUT.mkdir(exist_ok=True)
    if not (rank.OUT/'capacity_training.json').exists():
        audit.training_folds()
    report = json.loads((rank.OUT/'capacity_training.json').read_text())
    fold = np.asarray(report['folds'])[:,:,0]
    result = bool(fold[:,:,2].min()>0 and (fold[:,:,2].mean(0)>fold[:,:,1].mean(0)).all())
    (OUT/'results.json').write_text(json.dumps(dict(use_capacity=result,training_windows=[[9,12],[13,16]],
        complete=fold[:,:,1].tolist(),capacity=fold[:,:,2].tolist(),learned=fold[:,:,3].tolist(),
        coefficient_source='unchanged rank model trained through day 16'),indent=2))
    return result


def correction():
    nets,weight = load_models()
    return lambda raw:change(raw,nets,weight)


def layer():
    before,update = parent.layer(),correction()
    def predict(raw,initial):
        p = before(raw,initial).copy()
        p[:,:,2] += update(raw)
        return p
    return predict


def replay_with():
    update = correction()
    def read(d,raw):
        p,h,n = history.read(d,raw)
        p[:,:,2] += update(raw)
        h[:,:,2] += update(raw[:n])
        return p,h,n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
