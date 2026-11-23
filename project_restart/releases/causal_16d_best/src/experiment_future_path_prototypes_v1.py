"""Train matched regression and distributional heads for future 21-60 tick return paths."""
import copy
import json
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from torch import nn
from torch.nn import functional as F
import experiment_consensus_horizon_portfolios_v1 as blend
import experiment_pairwise_book_tree_v1 as book
import experiment_transitory_price_heads_v1 as frozen
import experiment_auxiliary_quote_decoder_v1 as decoder
import experiment_tail_portfolio_filter_v1 as panel
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, corr
from experiment_predictive_book_modes_v1 import corrections, slopes
from experiment_book_transition_rewards_v1 import replay

OUT = ART/'future_path_prototypes_v1'
EDGES = (20, 30, 40, 50, 60)
K = 32


def data(end, encoder):
    blocks = [[], [], []]
    for d in range(end):
        raw = book.rawday(d)
        ticks = np.arange(d%7, len(raw)-61, 7)
        x, v = frozen.encoding(raw, ticks, encoder, d, fixed_batch=True)
        price = (raw[:, :, 0].astype(float)-.00136).cumsum(0)
        path = np.diff(np.stack([price[ticks+h] for h in EDGES], axis=2), axis=2)*.161
        official = labels(d).reshape(-1,50,3)[ticks]
        tail = official[:, :, 2] - panel.RATIO*official[:, :, 1]
        # Enforce the official terminal payout while retaining the four observed increments.
        path[:, :, -1] += tail-path.sum(2)
        for dest, arr in zip(blocks,(x,v,torch.from_numpy(path.astype('float32')).cuda())):
            dest.append(arr)
    x, v, path = map(torch.cat, blocks)
    market = path.mean(1, keepdim=True)
    local = (path-market)/v[:, :, None]
    common = market/v.mean(1)[:, None, None]
    return x, v, torch.cat((local,common),dim=1)


def dictionary(y):
    rng = np.random.default_rng(9085)
    centers, payouts = [], []
    for block in (y[:, :50], y[:, 50:]):
        sample = block.reshape(-1,4).cpu().numpy()
        sample = sample[rng.choice(len(sample), min(len(sample),50000), replace=False)]
        # Cluster signed path shapes; fit the return scale with unclipped training payouts.
        path = np.cumsum(sample, axis=1)
        scale = np.maximum(np.std(path, axis=0), .04)
        z = np.arcsinh(path/scale).astype('float32')
        km = MiniBatchKMeans(n_clusters=K,batch_size=1024,n_init=3,random_state=389).fit(z)
        ids = km.predict(z)
        counts = np.bincount(ids,minlength=K)
        centroid = np.stack([np.bincount(ids,weights=sample[:, k],minlength=K)/(counts+.01) for k in range(4)],axis=1)
        centers.append((torch.from_numpy(km.cluster_centers_).cuda(),torch.from_numpy(scale).cuda()))
        payouts.append(torch.from_numpy(centroid.astype('float32')).cuda())
    return centers,payouts


class Net(nn.Module):
    def __init__(self,width,payouts,distributed):
        super().__init__()
        self.distributed = distributed
        self.enc = nn.Sequential(nn.Linear(width,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.15),nn.Linear(64,40),nn.SiLU())
        n = K if distributed else 4
        self.local = nn.Sequential(nn.Linear(80,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,n))
        self.market = nn.Sequential(nn.Linear(80,48),nn.LayerNorm(48),nn.SiLU(),nn.Dropout(.12),nn.Linear(48,n))
        self.register_buffer('payout',torch.stack(payouts))

    def forward(self,x):
        h = self.enc(x.float())
        m = h.mean(1,keepdim=True)
        a = self.local(torch.cat((h,m.expand_as(h)),dim=2)).float()
        b = self.market(torch.cat((m[:,0],h.std(1,unbiased=False)),dim=1)).float()[:,None]
        if self.distributed:
            path = torch.cat((a.softmax(2)@self.payout[0],b.softmax(2)@self.payout[1]),dim=1)
        else:
            path = torch.cat((a,b),dim=1)
        return path, a, b


def train(dataset):
    x,v,y = dataset
    centers,payouts = dictionary(y)
    ids = []
    for block,(c,scale) in zip((y[:,:50],y[:,50:]),centers):
        z = (block.cumsum(2)/scale).asinh()
        wins = []
        for u in z.split(96):
            dist = c.square().sum(1) - 2*u@c.T
            wins.append(dist.argmin(2))
        ids.append(torch.cat(wins))
    trained = []
    for distributed in (False,True):
        torch.manual_seed(8441)
        net = Net(x.shape[2],payouts,distributed).cuda()
        ema = copy.deepcopy(net).eval()
        opt = torch.optim.AdamW(net.parameters(),lr=.00075,weight_decay=.035)
        for ep in range(11):
            for batch in torch.randperm(len(x),device='cuda').split(64):
                with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
                    path,a,b = net(x[batch])
                target = y[batch]
                loss = 0.
                for weight,part in ((1.,slice(0,50)),(.62,slice(50,51))):
                    pred = path[:,part]
                    truth = target[:,part]
                    loss += weight*(F.smooth_l1_loss(pred.sum(2),truth.sum(2),beta=.7)+.5*F.smooth_l1_loss(pred,truth,beta=.35))
                if distributed:
                    loss += .14*(F.cross_entropy(a.reshape(-1,K),ids[0][batch].ravel())+
                                 .62*F.cross_entropy(b.reshape(-1,K),ids[1][batch].ravel()))
                opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(),.8)
                opt.step()
                with torch.no_grad():
                    torch._foreach_lerp_(list(ema.parameters()),list(net.parameters()),.024)
            if ep in (4,8):
                for g in opt.param_groups:g['lr'] *= .55
        trained.append(ema)
    return trained


@torch.inference_mode()
def side(x,v,model):
    outputs = []
    for k in range(0,len(x),64):
        path = model(x[k:k+64])[0].sum(2)
        rv = v[k:k+64]
        rel = path[:,:50]*rv
        outputs.append((rel-rel.mean(1,keepdim=True)+path[:,50:]*rv.mean(1,keepdim=True)).cpu().numpy())
    return np.concatenate(outputs)


def run():
    OUT.mkdir(exist_ok=True)
    hist = [[],[]]
    prior = []
    daily = []
    folds = []
    for end,stop in ((8,12),(12,14),(14,16)):
        encoder = decoder.net(end)
        dataset = data(end,encoder)
        networks = train(dataset)
        del dataset
        c = blend.fixed_weight(prior)
        w = [.65*np.median(h,axis=0) if h else np.array([.10,.10]) for h in hist]
        groups = []
        for d in range(end,stop):
            p,y,fs,prev = blend.training_day(d,prior)
            prior.append(prev)
            raw = book.rawday(d)
            t = np.arange(d%4,len(raw),4)
            p = blend.parent.fixed_forecast(p,fs,c)[t]
            y = y[t]
            x,v = frozen.encoding(raw,t,encoder,fixed_batch=True)
            small = t<195
            if d==end:
                xx,vv = frozen.encoding(raw[:195],t[small],encoder,fixed_batch=True)
            zs = []
            preds = []
            for i,model in enumerate(networks):
                n = side(x,v,model)
                if d==end:np.testing.assert_allclose(n[small],side(xx,vv,model),atol=.0001,rtol=0)
                z = corrections(p,n)
                zs.append(z@w[i])
                preds.append(n)
                hist[i].append(slopes(p,y,z))
            base = panel.scores(p,y,0)
            daily.append(panel.scores(p,y,zs[1])-base)
            groups.append((p,y,*zs,*preds))
        p,y,direct,proto,dnet,pnet = map(np.concatenate,zip(*groups))
        tail = (y[:,:,2]-panel.RATIO*y[:,:,1]).reshape(-1,1)
        entry = dict(days=[end+1,stop],baseline=panel.scores(p,y,0).tolist(),
                     direct=panel.scores(p,y,direct).tolist(),distribution=panel.scores(p,y,proto).tolist(),
                     standalone_tail=[float(corr(s.reshape(-1,1),tail)[0]) for s in (dnet,pnet)])
        folds.append(entry)
        print(entry,flush=True)
        (OUT/'results.json').write_text(json.dumps(dict(folds=folds),indent=2))
    gain = np.stack([np.subtract(f['distribution'],f['baseline']) for f in folds])
    advantage = np.stack([np.subtract(f['distribution'],f['direct']) for f in folds])
    gate = bool((gain[-2:].min(0)>0).all() and (advantage.mean(0)>0).all() and ((np.stack(daily)>0).sum(0)>=5).all())
    (OUT/'results.json').write_text(json.dumps(dict(folds=folds,retained=gate),indent=2))
    if gate:
        encoder = decoder.net(16)
        dataset = data(16,encoder)
        model = train(dataset)[1]
        del dataset
        w = .65*np.median(hist[1],axis=0)
        torch.save(model.state_dict(),OUT/'net_16.pth')
        np.save(OUT/'weights.npy',w)
        def predict(raw,p):
            q=p.copy()
            for offset in range(4):
                t=np.arange(offset,len(p),4)
                x,v=frozen.encoding(raw,t,encoder,fixed_batch=True)
                q[t,:,2]+=corrections(p[t],side(x,v,model))@w
            return q
        result = replay(predict)
        (OUT/'raw_replay.json').write_text(json.dumps(result,indent=2))
        print(result,flush=True)


if __name__ == '__main__':
    run()
