"""Forecast how much of a dominant relative move survives its subsequent ten-tick blocks."""
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_dominant_far_block_v1 as base

OUT=base.peer.ART/'far_shock_followthrough_v1'
SEED=47293


def targets(path, control):
    slot=path.abs().argmax(-1,keepdim=True)
    move=path.gather(2,slot).squeeze(2)
    order=torch.arange(4,device=path.device).expand_as(path)
    if control:
        # Reverse the order of the three other blocks, keeping the dominant block in its original slot.
        rank=order-(order>slot).long()
        rev=2-rank
        ids=torch.where(order==slot,slot,rev+(rev>=slot).long())
        path=path.gather(2,ids)
    before=(path*(order<slot)).sum(2)
    after=(path*(order>slot)).sum(2)
    sign=(move>0).float()+.5*(move==0)
    orientation=torch.where(move<0,-1.,1.)
    post=after*orientation
    torch.testing.assert_close(before+move+orientation*post,path.sum(2),rtol=1e-5,atol=2e-6)
    return before,sign,move.abs(),post


class Head(base.Head):
    def __init__(self,width,joint=False):
        super().__init__(width)
        self.joint=joint
        self.read[-1]=nn.Linear(48,11 if joint else 10)

    def forward(self,x):
        h=self.enc(x.float())
        z=self.read(torch.cat((h,h.mean(1,keepdim=True).expand_as(h)),2))
        before,sign=z[:,:,0],z[:,:,1]
        # Each dominant-move direction has a size, two post-move amounts, and a continuation probability.
        v=z[:,:,2:10].reshape(*z.shape[:2],2,4)
        mag,up,down=F.softplus(v[:,:,:,:3]).unbind(3)
        logit=v[:,:,:,3]
        prob=logit.sigmoid()
        net=mag+prob*up-(1-prob)*down
        p=sign.sigmoid()
        far=before+p*net[:,:,1]-(1-p)*net[:,:,0]
        parts=(before,sign,mag,up,down,logit)
        if self.joint:
            near=z[:,:,10]
            return near+far, parts, near, far
        return far, parts


def train(end,enc,seed=SEED,joint=False):
    x,risk,y=base.paths.data(end,enc)
    near=None
    if joint:
        observed=[]
        for d in range(end):
            labels=base.peer.labels(d).reshape(-1,50,3)
            t=np.arange(d%7,len(labels)-61,7)
            val=base.RATIO*labels[t,:,1]
            observed.append(torch.from_numpy((val-val.mean(1,keepdims=True)).astype('float32')).cuda())
        near=torch.cat(observed)/risk
    path=y[:,:50].contiguous()
    truth=path.sum(2)
    models=[]
    for control in (True,False):
        before,sign,mag,post=targets(path,control)
        torch.manual_seed(seed)
        model=Head(x.shape[2],joint).cuda()
        ema=copy.deepcopy(model).eval()
        optimizer=torch.optim.AdamW(model.parameters(),lr=.00065,weight_decay=.04)
        for epoch in range(9):
            for i in torch.randperm(len(x),device='cuda').split(64):
                output=model(x[i])
                total,(pre,direction,size,up,down,follow)=output[:2]
                if joint:
                    loss=F.smooth_l1_loss(total,near[i]+truth[i],beta=.9)
                    loss+=.55*F.smooth_l1_loss(output[3],truth[i],beta=.9)+.45*F.smooth_l1_loss(output[2],near[i],beta=.7)
                else:
                    loss=F.smooth_l1_loss(total,truth[i],beta=.9)
                loss+=.28*(F.smooth_l1_loss(pre,before[i],beta=.70)+F.binary_cross_entropy_with_logits(direction,sign[i]))
                s=(sign[i]>=.5).long().unsqueeze(2)
                size,up,down,follow=[q.gather(2,s).squeeze(2) for q in (size,up,down,follow)]
                outcome=post[i]
                # A zero post-move amount is assigned to the continuation branch with magnitude zero.
                hit=(outcome>=0).float()
                value=torch.where(hit.bool(),up,down)
                loss+=.24*(F.smooth_l1_loss(size,mag[i],beta=.60)+F.smooth_l1_loss(value,outcome.abs(),beta=.60)
                           +F.binary_cross_entropy_with_logits(follow,hit))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(),.8)
                optimizer.step()
                with torch.no_grad():torch._foreach_lerp_(list(ema.parameters()),list(model.parameters()),.024)
            if epoch in (4,7):
                for group in optimizer.param_groups:group['lr']*=.55
        models.append(ema)
    return models


def load_heads(out,constructor,transform=None):
    parent=base.load()
    enc=base.paths.decoder.net(16)
    models=[]
    for i in (0,1):
        state=torch.load(out/f'head_{i}.pth',map_location='cpu',weights_only=True)
        net=constructor(state['enc.0.weight'].shape[1],i).cuda().eval()
        net.load_state_dict(state);models.append(net)
    def predict(raw,p):
        q=parent(raw,p).copy()
        z=base.delta(raw,np.arange(len(raw)),enc,models)
        q[:,:,2]+=z if transform is None else transform(q,z)
        return q
    return predict


def load(out=OUT,joint=False):
    return load_heads(out,lambda width,i: Head(width,joint))


def replay_candidate(out,factory,baseline=None):
    reference = base.OUT if baseline is None else baseline
    best=json.loads((reference/'reload_verification.json').read_text())['pooled']['score']
    first=None
    for name in ('raw_replay','reload_verification'):
        r=base.peer.walk.replay(factory())
        if baseline is not None:
            r['parent_comparison'] = dict(model=reference.name, score=best,
                                          gain=r['pooled']['score']-best)
        if first is not None:
            np.testing.assert_allclose(r['pooled']['candidate'],first['pooled']['candidate'],rtol=0,atol=1e-10)
            for d in r['daily']:
                assert abs(first['daily'][d]['score']-r['daily'][d]['score'])<1e-10
        (out/f'{name}.json').write_text(json.dumps(r,indent=2))
        print(name,r['pooled']['score'],flush=True)
        if r['pooled']['score']<=best:break
        first=r


def replay(out=OUT,joint=False):
    replay_candidate(out,lambda: load(out,joint))


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(2)
    if base.screen(OUT,SEED,train):replay()
