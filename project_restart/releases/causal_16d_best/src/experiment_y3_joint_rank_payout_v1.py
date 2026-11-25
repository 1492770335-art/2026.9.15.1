"""Forecast the cross-sectional assignment of names to return bins and the ordered payoff of each bin."""
import copy
import json
import sys

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_y3_disjoint_stock_views_v1 as inputs
import y3_protected_path_references as history

parent,base = history.incumbent,history.base
OUT = parent.OUT.parent/'y3_joint_rank_payout_v1'
BINS = 10
ARMS = ('unchanged','direct far','independent far','balanced far','balanced Y3','balanced minus independent',
        'balanced minus direct','balanced probability contribution','balanced payout contribution',
        'relative balance contribution','relative direct contrast','early balanced contribution','combined protection')


def samples(d):
    raw = history.trial.clock.book.rawday(d)
    y = history.trial.clock.labels(d).reshape(-1,50,3)
    t = np.concatenate([np.arange(lo+(d*5)%10,hi-60,10) for lo,hi in base.segments(len(raw))])
    x,ph = inputs.features(raw,t)
    early = y[t,:,0]*.160943/.558942
    near = base.RATIO*y[t,:,1]
    target = np.stack((early,near-early,y[t,:,2]-near),2).astype('float32')
    return x,ph,target


def ordinal(y):
    ordered = y.sort(dim=1).values
    pay = ordered.reshape(len(y),BINS,5,3).mean(2).transpose(1,2)
    # Give exactly equal outcomes identical soft assignments when they straddle a bin boundary.
    values = y[:,:,:,None]
    sorted_values = ordered.transpose(1,2)[:,None]
    equal = (values==sorted_values).float().reshape(len(y),50,3,BINS,5).sum(-1)
    prob = equal/equal.sum(-1,keepdim=True)
    return prob,pay


class Model(nn.Module):
    def __init__(self,width,mode):
        super().__init__()
        self.mode = mode
        self.encode = nn.Sequential(nn.Linear(width,64),nn.LayerNorm(64),nn.SiLU(),nn.Linear(64,40),nn.SiLU())
        self.global_net = nn.Sequential(nn.Linear(83,56),nn.LayerNorm(56),nn.SiLU(),nn.Dropout(.10))
        self.local_net = nn.Sequential(nn.Linear(96,56),nn.LayerNorm(56),nn.SiLU(),nn.Dropout(.12))
        self.direct = nn.Linear(56,3)
        self.logits = nn.Linear(56,3*BINS)
        self.gaps = nn.Linear(56,3*BINS)
        nn.init.constant_(self.gaps.bias,-2.)
        self.mid = nn.Linear(56,3)

    def forward(self,x,phase):
        h = self.encode(x.float())
        g = self.global_net(torch.cat((h.mean(1),h.std(1,unbiased=False),phase),1))
        local = self.local_net(torch.cat((h,g[:,None].expand(-1,50,-1)),2))
        raw = self.direct(local)
        if self.mode==0:
            return raw,None,None
        logits = self.logits(local).reshape(len(x),50,3,BINS)
        probs = logits.softmax(-1)
        if self.mode==2:
            for _ in range(4):
                probs = probs*(5/probs.sum(1,keepdim=True).clamp_min(1e-9))
                probs = probs/probs.sum(-1,keepdim=True).clamp_min(1e-9)
        gaps = F.softplus(self.gaps(g).reshape(len(x),3,BINS)).cumsum(-1)
        pay = gaps-gaps.mean(-1,keepdim=True)+self.mid(g)[:,:,None]
        pred = (probs*pay[:,None]).sum(-1)
        return pred,probs,pay


def fit(rows):
    x,ph,y = [torch.from_numpy(np.concatenate(c)).cuda() for c in zip(*rows)]
    models = []
    for mode in (0,1,2):
        torch.manual_seed(41237)
        net = Model(x.shape[-1],mode).cuda()
        opt = torch.optim.AdamW(net.parameters(),lr=.0009,weight_decay=.045)
        ema = copy.deepcopy(net).eval()
        for epoch in range(7):
            if epoch in (4,6):
                opt.param_groups[0]['lr'] *= .55
            for ids in torch.randperm(len(y),device='cuda').split(48):
                target = y[ids]
                pred,assignment,payout = net(x[ids],ph[ids])
                loss = F.smooth_l1_loss(pred,target,beta=.45)+.6*F.smooth_l1_loss(pred.sum(2),target.sum(2),beta=.75)
                if mode:
                    truth,ordered = ordinal(target)
                    loss += .45*F.smooth_l1_loss(payout,ordered,beta=.3)
                    loss -= .40*(truth*assignment.clamp_min(1e-8).log()).sum(-1).mean()
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),1.)
                opt.step()
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.05)
        models.append(ema)
    return models


@torch.inference_mode()
def signals(raw,models):
    x,phase = inputs.features(raw,np.arange(len(raw)))
    rows = []
    for i in range(0,len(raw),96):
        v,ph = torch.from_numpy(x[i:i+96]).cuda(),torch.from_numpy(phase[i:i+96]).cuda()
        simple = models[0](v,ph)[0]
        free,a,p = models[1](v,ph)
        balanced,b,q = models[2](v,ph)
        # Symmetric two-factor decomposition is exact, including the interaction term.
        assignment = ((b-a)*.5*(q+p)[:,None]).sum(-1)
        payout = ((b+a)*.5*(q-p)[:,None]).sum(-1)
        torch.testing.assert_close(assignment+payout,balanced-free,atol=3e-6,rtol=0)
        def relative(z):
            return z-z.mean(1,keepdim=True)
        far = balanced[:,:,2]
        direct = simple[:,:,2]
        ordinary = free[:,:,2]
        diff = far-ordinary
        early = relative(balanced[:,:,0]-free[:,:,0])
        rows.append(torch.stack((direct,ordinary,far,balanced.sum(2),diff,far-direct,
            assignment[:,:,2],payout[:,:,2],relative(diff),relative(far-direct),early,relative(diff)+early),2).cpu().numpy())
    return np.concatenate(rows)


def train():
    OUT.mkdir(exist_ok=True)
    rows,records,folds,daily,audit = [],[],[],[],[]
    for d in range(16):
        if d in (4,8,12):
            models = fit(rows)
            print('fitted through',d,flush=True)
            if d>=8:
                w = base.calibrate(records)
                fold = []
        if d>=4:
            raw = history.trial.clock.book.rawday(d)
            y = history.trial.clock.labels(d).reshape(-1,50,3)
            z = signals(raw,models)
            if d==4:
                np.testing.assert_allclose(z[:133],signals(raw[:133],models),atol=3e-6,rtol=0)
            if d>=8:
                ps = history.references(d,raw)
            else:
                ps = history.previous.incumbent.parent.warm.read(d,raw)
            pair = [base.records(p,y,z) for p in ps]
            records.append(pair)
            far = y[:,:,2]-base.RATIO*y[:,:,1]
            audit.append([float(np.corrcoef(z[:,:,k].ravel(),far.ravel())[0,1]) for k in range(3)])
            if d>=8:
                fold.append(pair)
                daily.append([base.score(m,w) for m in pair])
                if d in (11,15):
                    folds.append([base.score(m,w) for m in np.sum(fold,0)])
        rows.append(samples(d))
    f,ds = np.asarray(folds)[:,:,0],np.asarray(daily)[:,:,0]
    good = [j for j in range(3,len(ARMS)) if f[:,:,j].min()>0 and (ds[:,:,j]>0).sum(0).min()>=5 and
            (f[:,:,j].mean(0)>f[:,:,1:3].mean(0).max(1)).all()]
    pick = max(good,key=lambda j:f[:,:,j].min()) if good else 0
    (OUT/'results.json').write_text(json.dumps(dict(selected=pick,arms=ARMS,daily_standalone_far=audit,
        folds=np.asarray(folds).tolist(),daily=np.asarray(daily).tolist()),indent=2))
    print('selected',ARMS[pick],flush=True)
    if pick:
        models = fit(rows)
        params = {str(i)+'_'+k:v.cpu().numpy() for i,m in enumerate(models) for k,v in m.state_dict().items()}
        np.savez(OUT/'model.npz',width=rows[0][0].shape[2],w=base.calibrate(records)[pick-1],pick=pick-1,**params)
    return bool(pick)


def correction():
    with np.load(OUT/'model.npz') as f:
        nets = [Model(int(f['width']),i).cuda().eval() for i in range(3)]
        for i,net in enumerate(nets):
            net.load_state_dict({k:torch.from_numpy(f[str(i)+'_'+k]) for k in net.state_dict()})
        w,pick = float(f['w']),int(f['pick'])
    return lambda raw:(signals(raw,nets)[:,:,pick]*w).astype('float32')


def layer():
    previous,update = parent.layer(),correction()
    def predict(raw,initial):
        p = previous(raw,initial).copy()
        p[:,:,2] += update(raw)
        return p
    return predict


def replay_with():
    update = correction()
    def read(day,raw):
        p,h,n = history.read(day,raw)
        p[:,:,2] += update(raw)
        h[:,:,2] += update(raw[:n])
        return p,h,n
    return history.stack.verifier,read


if __name__=='__main__' and (sys.argv[1:]==['replay'] or train()):
    base.replay.replay(sys.modules[__name__])
