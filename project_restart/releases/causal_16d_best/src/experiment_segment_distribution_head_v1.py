"""Estimate the far return from the histogram of four consecutive ten-tick payouts."""
import copy
import json
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_future_path_prototypes_v1 as paths
import experiment_rebased_expert_mixer_v1 as best
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'segment_distribution_head_v1'
CUTS=(.04,.17,.33,.50,.67,.83,.96)
N=len(CUTS)+1


def histogram(y):
    # Separate grids for the market portfolio and the stock-relative returns.
    counts=[];centers=[]
    for a in (y[:,:50],y[:,50:]):
        v=a.cpu().numpy()
        cut=np.quantile(v[::3],CUTS).astype('float32')
        ids=torch.bucketize(a.contiguous(),torch.from_numpy(cut).cuda())
        hot=F.one_hot(ids,N)
        cnt=hot.sum((0,1,2))
        payout=(hot*a[:,:,:,None]).sum((0,1,2))/cnt.clamp_min(1)
        counts.append(hot.float().mean(2));centers.append(payout)
    return torch.cat(counts,1),torch.stack(centers)


class Head(nn.Module):
    def __init__(self,width,payout):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(width,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,40),nn.SiLU())
        def outputs():
            return nn.Sequential(nn.Linear(80,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,1+N))
        self.local=outputs();self.market=outputs()
        self.register_buffer('payout',payout)

    def forward(self,x):
        h=self.enc(x.float());m=h.mean(1,keepdim=True)
        a=self.local(torch.cat((h,m.expand_as(h)),dim=2))
        b=self.market(torch.cat((m[:,0],h.std(1,unbiased=False)),dim=1))[:,None]
        z=torch.cat((a,b),1)
        logits=z[:,:,1:].float()
        loc=logits[:,:50].softmax(2)@self.payout[0]
        common=logits[:,50:].softmax(2)@self.payout[1]
        return z[:,:,0].float(),4*torch.cat((loc,common),1),logits


def train(ds, arms=(False, True), batch_schedule=None, seed=2942):
    x,v,path=ds
    soft,unit=histogram(path)
    truth=path.sum(2);models=[]
    for aux in arms:
        torch.manual_seed(seed)
        net=Head(x.shape[2],unit).cuda();avg=copy.deepcopy(net).eval()
        optimizer=torch.optim.AdamW(net.parameters(),lr=.00065,weight_decay=.04)
        for epoch in range(9):
            order = torch.randperm(len(x),device='cuda')
            batches = order.split(64) if batch_schedule is None else batch_schedule(order)
            for ids in batches:
                direct,quant,logits=net(x[ids]);b=truth[ids]
                loss=0.
                for s,f in ((slice(0,50),1.),(slice(50,51),.60)):
                    loss+=f*F.smooth_l1_loss(direct[:,s],b[:,s],beta=.78)
                    if aux:
                        ce=-(soft[ids,s]*logits[:,s].log_softmax(2)).sum(2).mean()
                        loss+=f*(.60*ce+.30*F.smooth_l1_loss(quant[:,s],b[:,s],beta=.78))
                optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),.8)
                optimizer.step()
                with torch.no_grad():torch._foreach_lerp_(list(avg.parameters()),list(net.parameters()),.024)
            if epoch in (4,7):
                for group in optimizer.param_groups:group['lr']*=.55
        models.append(avg)
    return models


@torch.inference_mode()
def sides(raw,t,encoder,nets):
    xx,v=paths.frozen.encoding(raw,t,encoder,fixed_batch=True)
    arr=[[],[]]
    for start in range(0,len(t),64):
        x=xx[start:start+64];scale=v[start:start+64]
        for j,net in enumerate(nets):
            a,b,_=net(x)
            forecast=(.50*a+.50*b) if j else a
            local=forecast[:,:50]*scale
            market=forecast[:,50:]*scale.mean(1,keepdim=True)
            arr[j].append((local-local.mean(1,keepdim=True)+market).cpu().numpy())
    return [np.concatenate(a) for a in arr]


def changes(p,a,b):
    delta=(b-a).astype(float);m=delta.mean(1,keepdims=True)
    z=delta-m;ref=p.astype(float)-p.mean(1,keepdims=True)
    gram=ref.transpose(0,2,1)@ref
    reg=np.eye(3)[None]*(.00004+.01*gram.diagonal(axis1=1,axis2=2))[:,None,:]
    novel=z-(ref@np.linalg.solve(gram+reg,ref.transpose(0,2,1)@z[:,:,None]))[:,:,0]
    return .15*m,.15*novel


def load(parent=None):
    if parent is None:parent=best.load()
    enc=paths.decoder.net(16);nets=[]
    for j in (0,1):
        st=torch.load(OUT/f'net_{j}.pth',map_location='cpu',weights_only=True)
        net=Head(st['enc.0.weight'].shape[1],st['payout']).cuda().eval()
        net.load_state_dict(st);nets.append(net)
    gate=np.load(OUT/'gate.npy')
    def predict(raw,p):
        q=parent(raw,p);v=changes(q,*sides(raw,np.arange(len(raw)),enc,nets))
        out=q.copy();out[:,:,2]+=sum(z*w for z,w in zip(v,gate))
        return out
    return predict


def main():
    OUT.mkdir(exist_ok=True);torch.set_num_threads(2);folds=[];daily=[];stand=[]
    for end,stop in ((8,12),(12,14),(14,16)):
        enc=paths.decoder.net(end)
        ds=paths.data(end,enc)
        nets=train(ds);del ds
        for begin in range(end,stop,2):
            groups=[[[],[]] for _ in range(2)];scores=[]
            for d in (begin,begin+1):
                raw=paths.book.rawday(d);t=np.arange(d%4,len(raw),4)
                y=labels(d).reshape(-1,50,3)[t]
                a,b=sides(raw,t,enc,nets)
                far=y[:,:,2]-(.160943/.272164)*y[:,:,1]
                scores.append((a,b,far))
                if d==begin:
                    tt=t[t<179];small=sides(raw[:179],tt,enc,nets)
                    for q,r in zip((a,b),small):
                        np.testing.assert_allclose(q[:len(tt)],r,rtol=0,atol=4e-5)
                for j,path in enumerate((best.refit.OUT/f'reference_{d}.npy',best.refit.core.OUT/f'pred_{d}.npy')):
                    p=np.load(path,mmap_mode='r')[t]
                    gains=[]
                    for k,v in enumerate(changes(p,a,b)):
                        q=p.copy();q[:,:,2]+=v
                        row=tuple(z.reshape(-1,3) for z in (p,q,y));groups[j][k].append(row)
                        sc=measure([row]);gains.append(sc['candidate'][2]-sc['base'][2])
                    if not j:daily.append(gains)
            folds.append([[measure(a) for a in pair] for pair in groups])
            a,b,far=map(np.concatenate,zip(*scores))
            stand.append([float(np.corrcoef(q.ravel(),far.ravel())[0,1]) for q in (a,b)])
            print(begin+1,'far',stand[-1],'delta',[[s['candidate'][2]-s['base'][2] for s in pair] for pair in folds[-1]],flush=True)
    g=np.array([[[s['candidate'][2]-s['base'][2] for s in pair] for pair in f] for f in folds])
    keep=(g[:,0].min(0)>1e-6)&(g[:,1].mean(0)>0)&((np.array(daily)>0).sum(0)>=5)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,kept=keep.tolist(),standalone_tail=stand,daily=daily),indent=2))
    print('retained',keep,flush=True)
    if keep.any():
        np.save(OUT/'gate.npy',keep)
        enc=paths.decoder.net(16);ds=paths.data(16,enc)
        for j,n in enumerate(train(ds)):torch.save(n.cpu().state_dict(),OUT/f'net_{j}.pth')
        del ds
        replay=best.roll.gls.best.walk.replay
        res=replay(load());(OUT/'raw_replay.json').write_text(json.dumps(res,indent=2))
        prev=json.loads((best.OUT/'reload_verification.json').read_text())['pooled']
        print('official',res['pooled']['score'],res['pooled']['score']-prev['score'],flush=True)
        if res['pooled']['score']>prev['score']:
            check=replay(load())
            np.testing.assert_allclose(check['pooled']['candidate'],res['pooled']['candidate'],rtol=0,atol=1e-10)
            (OUT/'reload_verification.json').write_text(json.dumps(check,indent=2))


if __name__=='__main__':main()
