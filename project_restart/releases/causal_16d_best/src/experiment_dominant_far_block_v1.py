"""Compare forecasting the dominant idiosyncratic ten-tick move with a matched random ten-tick move."""
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_future_path_prototypes_v1 as paths
import experiment_pressure_seed_average_v1 as incumbent
import experiment_y3_error_components_v1 as audit

peer = audit.peer
OUT = peer.ART/'dominant_far_block_v1'
RATIO = audit.SCALE[2]/audit.SCALE[1]
SEED = 18317


class Head(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(width,64), nn.LayerNorm(64), nn.SiLU(), nn.Dropout(.13),
                                 nn.Linear(64,40), nn.SiLU())
        self.read = nn.Sequential(nn.Linear(80,48), nn.LayerNorm(48), nn.SiLU(), nn.Linear(48,4))

    def forward(self, x):
        h = self.enc(x.float())
        z = self.read(torch.cat((h, h.mean(1,keepdim=True).expand_as(h)),2))
        logit, positive, negative, remainder = z.unbind(2)
        a, b = F.softplus(positive), F.softplus(negative)
        prob = logit.sigmoid()
        return prob*a-(1-prob)*b + remainder, logit, a, b, remainder


def train(end, enc, seed=SEED):
    x, _, path = paths.data(end,enc)
    path = path[:,:50].contiguous()
    generator = torch.Generator(device='cuda').manual_seed(SEED+1)
    dominant = path.abs().argmax(2,keepdim=True)
    # Shuffle the chosen block among stocks at each origin; match its time-position frequencies.
    perm = torch.rand(path.shape[:2],generator=generator,device='cuda').argsort(1)
    placebo = dominant.gather(1,perm[:,:,None])
    truth = path.sum(2)
    moves = [path.gather(2,slot)[:,:,0] for slot in (placebo,dominant)]
    return fit_targets(x, truth, moves, seed)


def fit_targets(x, truth, moves, seed):
    nets = []
    for chosen in moves:
        rest = truth-chosen
        sign = (chosen>0).float()+.5*(chosen==0)
        torch.manual_seed(seed)
        net = Head(x.shape[2]).cuda()
        avg = copy.deepcopy(net).eval()
        opt = torch.optim.AdamW(net.parameters(),lr=.00065,weight_decay=.04)
        for epoch in range(9):
            for i in torch.randperm(len(x),device='cuda').split(64):
                total, logits, a, b, remain = net(x[i])
                pos=sign[i]
                mag=chosen[i].abs()
                conditional=pos*F.smooth_l1_loss(a,mag,beta=.60,reduction='none')
                conditional+=(1-pos)*F.smooth_l1_loss(b,mag,beta=.60,reduction='none')
                loss=F.smooth_l1_loss(total,truth[i],beta=.9)+.40*F.binary_cross_entropy_with_logits(logits,pos)
                loss+=.38*(conditional.mean()+F.smooth_l1_loss(remain,rest[i],beta=.72))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),.8)
                opt.step()
                with torch.no_grad():
                    torch._foreach_lerp_(list(avg.parameters()),list(net.parameters()),.024)
            if epoch in (4,7):
                for g in opt.param_groups:g['lr']*=.55
        nets.append(avg)
    return nets


@torch.inference_mode()
def delta(raw,t,enc,nets):
    x,v = paths.frozen.encoding(raw,t,enc,fixed_batch=True)
    out = []
    for start in range(0,len(t),64):
        sl = slice(start,start+64)
        a,b = (net(x[sl])[0] for net in nets)
        rel = .15*(b-a)*v[sl]
        out.append((rel-rel.mean(1,keepdim=True)).cpu().numpy())
    return np.concatenate(out)


def gain(rows):
    p,y,z = (np.concatenate(a) for a in zip(*rows))
    def corr(a,b):
        return float(np.corrcoef(a.ravel(),b.ravel())[0,1])
    a,b = (v[:,:,2]-RATIO*v[:,:,1] for v in (p,y))
    a,b = (v-v.mean(1,keepdims=True) for v in (a,b))
    return dict(y3=corr(p[:,:,2]+z,y[:,:,2])-corr(p[:,:,2],y[:,:,2]),relative_far=corr(a+z,b)-corr(a,b))


def screen(out=OUT, seed=SEED, trainer=train):
    blocks,days = [],[]
    for end,stop in ((8,12),(12,14),(14,16)):
        enc=paths.decoder.net(end)
        nets=trainer(end,enc,seed)
        for begin in range(end,stop,2):
            rows=[]
            for d in (begin,begin+1):
                raw=paths.book.rawday(d)
                t=np.arange(d%4,len(raw),4)
                z=delta(raw,t,enc,nets)
                if d==begin:
                    st=t[t<213]
                    np.testing.assert_allclose(delta(raw[:213],st,enc,nets),z[:len(st)],rtol=0,atol=4e-5)
                rows.append((peer.reference(d)[t],peer.labels(d).reshape(-1,50,3)[t],z))
                days.append(gain(rows[-1:]))
            blocks.append(gain(rows))
            print(begin+1,blocks[-1],flush=True)
    keep=min(b['y3'] for b in blocks)>0 and sum(d['y3']>0 for d in days)>=5 and np.mean([b['relative_far'] for b in blocks])>0
    report=dict(folds=blocks,daily=days,retained=bool(keep),seed=seed)
    if trainer is train:
        report['assignment_seed']=SEED+1
    (out/'results.json').write_text(json.dumps(report,indent=2))
    if keep:
        for i,m in enumerate(trainer(16,paths.decoder.net(16),seed)):
            torch.save(m.cpu().state_dict(),out/f'head_{i}.pth')
    return keep


def load():
    parent=incumbent.load()
    enc=paths.decoder.net(16)
    nets=[]
    for i in (0,1):
        st=torch.load(OUT/f'head_{i}.pth',map_location='cpu',weights_only=True)
        m=Head(st['enc.0.weight'].shape[1]).cuda().eval()
        m.load_state_dict(st);nets.append(m)
    def predict(raw,p):
        out=parent(raw,p).copy()
        out[:,:,2]+=delta(raw,np.arange(len(raw)),enc,nets)
        return out
    return predict


def main():
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    if not screen():return
    best=json.loads((incumbent.OUT/'reload_verification.json').read_text())['pooled']['score']
    first=None
    for name in ('raw_replay','reload_verification'):
        res=peer.walk.replay(load())
        if first is not None:
            np.testing.assert_allclose(res['pooled']['candidate'],first['pooled']['candidate'],rtol=0,atol=1e-10)
            for d in res['daily']:
                assert abs(res['daily'][d]['score']-first['daily'][d]['score'])<1e-10
        (OUT/f'{name}.json').write_text(json.dumps(res,indent=2))
        print(name,res['pooled']['score'],flush=True)
        if res['pooled']['score']<=best:break
        first=res


if __name__=='__main__':main()
