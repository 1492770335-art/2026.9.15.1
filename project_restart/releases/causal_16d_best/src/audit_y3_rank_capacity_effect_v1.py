"""Separate probability changes caused at inference by column normalization and by learning with that constraint."""
import json

import numpy as np
import torch

import y3_rank_payout_references as history

model,base = history.incumbent,history.base


@torch.inference_mode()
def signals(raw,nets):
    x,ph = model.inputs.features(raw,np.arange(len(raw)))
    pieces = []
    for i in range(0,len(x),96):
        v,p = torch.from_numpy(x[i:i+96]).cuda(),torch.from_numpy(ph[i:i+96]).cuda()
        _,a,free = nets[0](v,p)
        _,b,trained = nets[1](v,p)
        balanced = a
        for _ in range(4):
            balanced = balanced*(5/balanced.sum(1,keepdim=True).clamp_min(1.e-9))
            balanced = balanced/balanced.sum(-1,keepdim=True).clamp_min(1.e-9)
        scale = .5*(free+trained)[:,None,:, :]
        mechanical = ((balanced-a)*scale).sum(-1)[:,:,2]
        learned = ((b-balanced)*scale).sum(-1)[:,:,2]
        full = ((b-a)*scale).sum(-1)[:,:,2]
        torch.testing.assert_close(mechanical+learned,full,atol=2e-6,rtol=0)
        pieces.append(torch.stack((full,mechanical,learned),2).cpu().numpy())
    return np.concatenate(pieces)


def training_folds():
    training,records,folds = [],[],[]
    data = json.loads((model.OUT/'results.json').read_text())
    idx = data['selected']
    for d in range(16):
        if d in (4,8,12):
            nets = model.fit(training)[1:]
            if d>=8:
                weight = base.calibrate(records)[0]
                fold = []
        if d>=4:
            raw = history.trial.clock.book.rawday(d)
            y = history.trial.clock.labels(d).reshape(-1,50,3)
            z = signals(raw,nets)
            ps = history.previous.references(d,raw) if d>=8 else history.previous.previous.incumbent.parent.warm.read(d,raw)
            pair = [base.records(p,y,z) for p in ps]
            records.append(pair)
            if d>=8:
                fold.append(pair)
                if d in (11,15):
                    f = np.asarray([base.score(m,np.full(3,weight)) for m in np.sum(fold,0)])
                    np.testing.assert_allclose(f[:,0,1],np.array(data['folds'])[(d-11)//4,:,0,idx],atol=2e-8,rtol=0)
                    folds.append(f.tolist())
        training.append(model.samples(d))
    with np.load(model.OUT/'model.npz') as f:
        np.testing.assert_allclose(f['w'],base.calibrate(records)[0],atol=1e-8,rtol=0)
    report = dict(arms=['unchanged','both','inference normalization only','training difference only'],folds=folds)
    (model.OUT/'capacity_training.json').write_text(json.dumps(report,indent=2))
    return folds


def main():
    folds = training_folds()
    with np.load(model.OUT/'model.npz') as f:
        weight = float(f['w'])
        nets = [model.Model(int(f['width']),idx).cuda().eval() for idx in (1,2)]
        for j,net in enumerate(nets,1):
            net.load_state_dict({k:torch.from_numpy(f[str(j)+'_'+k]) for k in net.state_dict()})
    outer = []
    for d in range(16,20):
        raw = history.trial.clock.book.rawday(d)
        y = history.trial.clock.labels(d).reshape(-1,50,3)
        p,_,_ = history.previous.read(d,raw)
        outer.append(base.records(p,y,signals(raw,nets)))
    replay = np.asarray(base.score(np.sum(outer,0),np.full(3,weight)))
    best = json.loads((model.OUT/'raw_replay.json').read_text())['pooled']
    assert abs(replay[0,1]-2*best['gain'])<2e-8
    out = dict(arms=['unchanged','both','inference normalization only','training difference only'],
               reference='frozen protected-path predecessor',metrics=['Y3 correlation gain','remote correlation gain'],
               folds=folds,outer=replay.tolist())
    (model.OUT/'capacity_ablation.json').write_text(json.dumps(out,indent=2))
    print('inner gains',np.asarray(folds)[:,:,0,:].round(6).tolist(),'outer',replay[0].round(7).tolist(),flush=True)


if __name__=='__main__':
    main()
