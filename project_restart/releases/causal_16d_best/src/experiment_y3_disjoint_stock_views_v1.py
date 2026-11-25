"""Train panel encoders on complementary stock groups and test whether cross-group payoffs transfer."""
import copy
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import y3_flow_exchange_references as history

base,parent = history.base,history.incumbent
OUT = parent.OUT.parent/'y3_disjoint_stock_views_v1'
MODES = ('same group','opposite group','opposite with consistency')
ARMS = ('unchanged','same group market','opposite group market','consistent market','opposite minus same',
        'consistent minus same','same group relative','opposite group relative','consistent relative',
        'relative opposite minus same','relative consistent minus same','opposite late market','consistent late market')
FIELDS = (0,10,11,16,21,25,34,37,39,40,45,52,54,55,60,64)


def features(raw,t):
    # All operations before the stock mask are independent for each stock.
    values = np.arcsinh(raw[:,:,FIELDS].astype('float32'))
    x = np.empty((len(t),50,5*len(FIELDS)),np.float16)
    for lo,hi in base.segments(len(raw)):
        ids = (t>=lo)&(t<hi)
        at = t[ids]-lo
        v = values[lo:hi]
        c = np.concatenate((np.zeros_like(v[:1]),np.cumsum(v,axis=0,dtype=float)))
        end = at+1
        history_windows = []
        for a,b in ((0,5),(5,20),(20,60),(60,180)):
            right,left = (end-a).clip(0),(end-b).clip(0)
            history_windows.append((c[right]-c[left])/(right-left).clip(1)[:,None,None])
        x[ids] = np.concatenate((v[at],*history_windows),2)
    session = np.where(t<2401,t,t-2401)
    phase = np.stack((np.exp(-session/240),session/2400,(t>=2401)),1).astype('float32')
    return x,phase


def records(day):
    raw = history.trial.clock.book.rawday(day)
    y = history.trial.clock.labels(day).reshape(-1,50,3)
    t = np.concatenate([np.arange(lo+(day*3)%8,hi-60,8) for lo,hi in base.segments(len(raw))])
    target = np.concatenate((y[t],(y[t,:,2]-base.RATIO*y[t,:,1])[:,:,None]),2)
    return *features(raw,t),target.astype('float32')


class GroupModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(5*len(FIELDS),56),nn.LayerNorm(56),nn.SiLU(),nn.Linear(56,32))
        self.head = nn.Sequential(nn.Linear(67,56),nn.LayerNorm(56),nn.SiLU(),
                                  nn.Dropout(.13),nn.Linear(56,4))
    def pools(self,h,phase,mask):
        mean = mask@h/25
        variance = mask@h.square()/25-mean.square()
        ph = phase.unsqueeze(1).expand(-1,mask.size(-2),-1)
        return self.head(torch.cat((mean,variance.clamp_min(1.e-5).sqrt(),ph),2))


def fit(rows):
    x,phase,y = [torch.from_numpy(np.concatenate(col)).cuda() for col in zip(*rows)]
    models = []
    for mode in range(3):
        torch.manual_seed(41861)
        net = GroupModel().cuda()
        opt = torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.05)
        ema = copy.deepcopy(net).eval()
        for epoch in range(6):
            if epoch in (3,5):
                opt.param_groups[0]['lr'] *= .55
            for idx in torch.randperm(len(x),device='cuda').split(48):
                partition = torch.rand((len(idx),50),device='cuda').argsort(1)<25
                mask = torch.stack((partition,~partition),1).float()
                h = net.encode(x[idx].float())
                preds = net.pools(h,phase[idx],mask)
                group = mask@y[idx]/25
                target = group if mode==0 else group.flip((1,))
                loss = F.smooth_l1_loss(preds,target,beta=.35)
                if mode == 2:
                    loss += .15*F.smooth_l1_loss(preds[:,0],preds[:,1],beta=.2)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.055)
        models.append(ema)
    return models


@torch.inference_mode()
def signals(raw,models):
    x,ph = features(raw,np.arange(len(raw)))
    rng = np.random.default_rng(9336)
    parts = rng.random((12,50)).argsort(1)<25
    masks = np.concatenate((parts,~parts))
    mask = torch.from_numpy(masks.astype('float32')).cuda()
    contrast = (mask[:12].T-.5)/6
    all_values = []
    for i in range(0,len(raw),96):
        v = torch.from_numpy(x[i:i+96]).cuda().float()
        phase = torch.from_numpy(ph[i:i+96]).cuda()
        common,relative,late = [],[],[]
        for m,net in enumerate(models):
            out = net.pools(net.encode(v),phase,mask.unsqueeze(0).expand(len(v),-1,-1))
            common.append(out[:,:,2].mean(1,keepdim=True).expand(-1,50))
            late.append(out[:,:,3].mean(1,keepdim=True).expand(-1,50))
            # Assign a half-panel estimate to the stock group it was trained to predict.
            diff = out[:,:12,2]-out[:,12:,2]
            relative.append((diff if m==0 else -diff)@contrast.T)
        a,b,c = common
        ar,br,cr = relative
        all_values.append(torch.stack((a,b,c,b-a,c-a,ar,br,cr,br-ar,cr-ar,late[1],late[2]),2).cpu().numpy())
    return np.concatenate(all_values)


def train():
    OUT.mkdir(exist_ok=True)
    rows,calibration,folds,daily = [],[],[],[]
    for d in range(16):
        if d in (4,8,12):
            nets = fit(rows)
            print('trained through day',d,flush=True)
            if d>=8:
                weight = base.calibrate(calibration)
                fold = []
        if d>=4:
            raw = history.trial.clock.book.rawday(d)
            z = signals(raw,nets)
            if d==4:
                np.testing.assert_allclose(z[:137],signals(raw[:137],nets),rtol=0,atol=3e-6)
                rng = np.random.default_rng(66)
                changed = raw[:137].copy()
                changed[:,:,0] = changed[:,rng.permutation(50),0]
                changed[:,:25] = raw[:137,:25]
                np.testing.assert_array_equal(features(changed,np.arange(137))[0][:,:25],
                                              features(raw[:137],np.arange(137))[0][:,:25])
            y = history.trial.clock.labels(d).reshape(-1,50,3)
            ps = history.references(d,raw) if d>=8 else parent.parent.warm.read(d,raw)
            pair = [base.records(p,y,z) for p in ps]
            calibration.append(pair)
            if d>=8:
                fold.append(pair)
                daily.append([base.score(v,weight) for v in pair])
                if d in (11,15):
                    folds.append([base.score(v,weight) for v in np.sum(fold,0)])
        rows.append(records(d))
    f,ds = np.asarray(folds)[:,:,0],np.asarray(daily)[:,:,0]
    ok = []
    for j in (2,3,4,5,7,8,9,10,11,12):
        control = 6 if 7<=j<=10 else 1
        if f[:,:,j].min()>0 and (ds[:,:,j]>0).sum(0).min()>=5 and (f[:,:,j].mean(0)>f[:,:,control].mean(0)).all():
            ok.append(j)
    pick = max(ok,key=lambda j:f[:,:,j].min()) if ok else 0
    (OUT/'results.json').write_text(json.dumps(dict(arms=ARMS,selected=pick,folds=np.asarray(folds).tolist(),
                                                    daily=np.asarray(daily).tolist()),indent=2))
    print('selected',ARMS[pick],flush=True)
    if pick:
        models = fit(rows)
        params = {str(i)+'_'+k:v.cpu().numpy() for i,m in enumerate(models) for k,v in m.state_dict().items()}
        np.savez(OUT/'model.npz',pick=pick-1,coef=base.calibrate(calibration)[pick-1],**params)
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        nets = [GroupModel().cuda().eval() for _ in MODES]
        for i,m in enumerate(nets):
            m.load_state_dict({key:torch.from_numpy(f[str(i)+'_'+key]) for key in m.state_dict()})
        pick,c = int(f['pick']),float(f['coef'])
    return lambda raw:(signals(raw,nets)[:,:,pick]*c).astype('float32')


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


if __name__ == '__main__' and (sys.argv[1:] == ['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
