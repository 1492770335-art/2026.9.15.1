"""Train three return heads with an auxiliary 12-step trajectory implied by their coefficients."""
import copy
import json
import numpy as np
import torch
from torch.nn import functional as F
import experiment_cross_date_response_shapes_v1 as linear
import experiment_block_error_whitening_v1 as block
from experiment_causal_cross_horizon_ridge_v1 import ART,labels,measure

OUT=ART/'response_template_training_v1'
WIDTHS=np.array([1,3,8],dtype='float32')
BASIS=np.zeros((12,3))
for k,(lo,hi) in enumerate(((0,1),(1,4),(4,12))):
    BASIS[lo:hi,k]=1/np.sqrt(hi-lo)


def templates(records):
    mats=[]
    for rec in zip(*records):
        grams,rhs,_=map(np.stack,zip(*rec))
        g=grams.mean(0)
        w=np.linalg.solve(grams+.8*np.diag(g.diagonal().clip(.0002)),rhs)
        v=linear.powers(g,.5)@w
        m=v.mean(0)
        dev=v-m
        # Products of response slopes from distinct training dates estimate the repeatable shape.
        cross=m.T@m-np.einsum('dfi,dfj->ij',dev,dev)/(len(v)*(len(v)-1))
        eig,u=np.linalg.eigh(cross)
        stable=(u*eig.clip(0,None))@u.T
        core=BASIS.T@stable@BASIS
        ridge=.5*np.diag(core.diagonal().clip(.001))+np.eye(3)*.002
        shape=BASIS.T+np.linalg.solve(core+ridge,BASIS.T@stable@(np.eye(12)-BASIS@BASIS.T))
        np.testing.assert_allclose(shape@BASIS,np.eye(3),atol=1e-8)
        mats.append(shape)
    return np.stack(mats).astype('float32')


def data(end,enc):
    features,returns=[],[]
    for d in range(end):
        raw=block.paths.book.rawday(d)
        t=np.arange(d%7,len(raw)-61,7)
        x,v=block.paths.frozen.encoding(raw,t,enc,d,fixed_batch=True)
        price=(raw[:,:,0].astype(float)-.00136).cumsum(0)
        fine=np.diff(np.stack([price[t+5*k] for k in range(13)],2),axis=2)
        levels=labels(d).reshape(-1,50,3)[t]/linear.SCALE
        for lo,hi,k in ((0,1,0),(1,4,1),(4,12,2)):
            band=levels[:,:,k]-(levels[:,:,k-1] if k else 0)
            fine[:,:,lo:hi]+=(band-fine[:,:,lo:hi].sum(2))[:,:,None]/(hi-lo)
        y=torch.from_numpy((fine*linear.SCALE[0]).astype('float32')).cuda()
        m=y.mean(1,keepdim=True)
        returns.append(torch.cat(((y-m)/v[:,:,None],m/v.mean(1)[:,None,None]),dim=1))
        features.append(x)
    return torch.cat(features),torch.cat(returns)


def levels(coef):
    width=coef.new_tensor(WIDTHS).sqrt()
    scale=coef.new_tensor(linear.SCALE/linear.SCALE[0])
    return (coef*width).cumsum(2)*scale


def train(end,enc,shapes):
    x,y=data(end,enc)
    basis=y.new_tensor(BASIS)
    truth=levels(y@basis)
    nets=[]
    for shape in (np.stack((BASIS.T,BASIS.T)),shapes):
        torch.manual_seed(5921)
        net=block.Head(x.shape[-1]).cuda()
        avg=copy.deepcopy(net).eval()
        opt=torch.optim.AdamW(net.parameters(),lr=.00065,weight_decay=.045)
        target_shape=y.new_tensor(shape)
        for epoch in range(9):
            for ix in torch.randperm(len(x),device='cuda').split(64):
                coef=net(x[ix])
                pred=levels(coef)
                loss=0.
                for k,(s,f) in enumerate(((slice(0,50),1.),(slice(50,51),.6))):
                    direct=F.smooth_l1_loss(pred[:,s],truth[ix,s],beta=.85)
                    curve=F.smooth_l1_loss(coef[:,s]@target_shape[k],y[ix,s],beta=.85)
                    loss+=f*(direct+.6*curve)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(),.8)
                opt.step()
                with torch.no_grad():torch._foreach_lerp_(list(avg.parameters()),list(net.parameters()),.024)
            if epoch in (4,7):
                for group in opt.param_groups:group['lr']*=.55
        nets.append(avg)
    return nets


@torch.inference_mode()
def signals(raw,t,enc,nets):
    x,v=block.paths.frozen.encoding(raw,t,enc,fixed_batch=True)
    rows=[]
    for start in range(0,len(t),64):
        sl=slice(start,start+64)
        step=levels(nets[1](x[sl])-nets[0](x[sl]))
        scale=torch.cat((v[sl],v[sl].mean(1,keepdim=True)),1)
        rows.append((step*scale[:,:,None]).cpu().numpy())
    q=np.concatenate(rows)
    local=q[:,:50]
    return local-local.mean(1,keepdims=True),q[:,50:]


def load():
    parent=linear.best.load()
    enc=block.paths.decoder.net(16)
    keep=np.load(OUT/'mask.npy')
    nets=[]
    for k in (0,1):
        s=torch.load(OUT/f'net_{k}.pth',map_location='cpu',weights_only=True)
        net=block.Head(s['enc.0.weight'].shape[1]).cuda().eval()
        net.load_state_dict(s)
        nets.append(net)
    def predict(raw,p):
        q=parent(raw,p).copy()
        for z,k in zip(signals(raw,np.arange(len(raw)),enc,nets),keep):
            q+=.15*z*k
        return q
    return predict


def main():
    torch.set_num_threads(2)
    OUT.mkdir(exist_ok=True)
    stats,folds,daily=[],[],[]
    for end,stop in ((8,12),(12,14),(14,16)):
        stats.extend(linear.record(d) for d in range(len(stats),end))
        shape=templates(stats)
        enc=block.paths.decoder.net(end)
        nets=train(end,enc,shape)
        for begin in range(end,stop,2):
            rows=[[],[]]
            for d in (begin,begin+1):
                raw=block.paths.book.rawday(d)
                t=np.arange(d%4,len(raw),4)
                y=labels(d).reshape(-1,50,3)[t]
                p=np.load(block.segment.best.refit.OUT/f'reference_{d}.npy',mmap_mode='r')[t].copy()
                p[:,:,1]=np.load(linear.earlier.OUT/f'y2_day_{d}.npy',mmap_mode='r')[t]
                sides=signals(raw,t,enc,nets)
                if d==begin:
                    early=t[t<181]
                    for a,b in zip(signals(raw[:181],early,enc,nets),sides):
                        np.testing.assert_allclose(a,b[:len(early)],atol=3e-5,rtol=0)
                for rs,side in zip(rows,sides):
                    rs.append(tuple(a.reshape(-1,3) for a in (p,p+.15*side,y)))
                daily.append([linear.gains(rs[-1:]) for rs in rows])
            folds.append([linear.gains(rs) for rs in rows])
            print(begin+1,'local/market IC gains',folds[-1],flush=True)
    keep=(np.min(folds,axis=0)>0)&((np.array(daily)>0).sum(0)>=5)
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,daily=daily,retained=keep.tolist()),indent=2))
    if keep.any():
        np.save(OUT/'mask.npy',keep)
        stats.extend(linear.record(d) for d in range(len(stats),16))
        enc=block.paths.decoder.net(16)
        for k,net in enumerate(train(16,enc,templates(stats))):
            torch.save(net.cpu().state_dict(),OUT/f'net_{k}.pth')
        base=json.loads((linear.best.OUT/'reload_verification.json').read_text())['pooled']['score']
        for name in ('raw_replay','reload_verification'):
            res=linear.best.online.best.roll.gls.best.walk.replay(load())
            if name=='reload_verification':
                old=json.loads((OUT/'raw_replay.json').read_text())
                np.testing.assert_allclose(res['pooled']['candidate'],old['pooled']['candidate'],atol=1e-10,rtol=0)
            (OUT/f'{name}.json').write_text(json.dumps(res,indent=2))
            print(name,res['pooled']['score'],flush=True)
            if res['pooled']['score']<=base:break


if __name__=='__main__':main()
