"""Matched FI-2010 versus in-domain temporal pretraining on chronological competition folds."""
import copy
import hashlib
import json
import zipfile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import experiment_peer_catchup_after_pulse_v1 as peer

OUT = peer.ART/'fi2010_transfer_v1'
ARCHIVE = OUT/'fi2010.zip'
SOURCE = 'https://raw.githubusercontent.com/zcakhaa/DeepLOB-Deep-Convolutional-Neural-Networks-for-Limit-Order-Books/master/data/data.zip'
LAGS = np.array([64,32,16,8,4,2,1,0])
ARMS = ('random','competition','external')
SEED = 32191
PRE_STEPS = 320
FIT_STEPS = 620


def pack(features,r,targets=None,training=True):
    r = r.astype('float32')
    cs = np.vstack((np.zeros((1,r.shape[1])),np.cumsum(r,axis=0,dtype=float)))
    sq = np.vstack((np.zeros((1,r.shape[1])),np.cumsum(r*r,axis=0,dtype=float)))
    end = np.arange(len(r))+1
    start = (end-96).clip(0)
    risk = np.sqrt((8+sq[end]-sq[start])/(8+end-start)[:,None]).clip(.20,8.)
    common = [np.arcsinh((cs[end]-cs[(end-k).clip(0)])/risk/np.sqrt(k)) for k in (1,5,20)]
    x = np.concatenate((features,np.stack(common,axis=-1)),axis=2).clip(-6,6).astype('float16')
    item = dict(x=torch.from_numpy(x).cuda())
    if training:
        t = np.arange(len(r)-60)
        ahead = []
        for low,high in ((0,5),(5,20),(20,60)):
            a,b = t+1+low,t+1+high
            ret = (cs[b]-cs[a])/(risk[t]*np.sqrt(high-low))
            rms = np.sqrt(np.maximum((sq[b]-sq[a])/(high-low),0.))/risk[t]
            ahead.extend((np.arcsinh(ret),np.log1p(rms)))
        item['proxy'] = torch.from_numpy(np.stack(ahead,2).astype('float32')).cuda()
    if targets is not None:
        item['y'] = torch.from_numpy(targets.copy()).cuda()
    return item


def competition(raw,targets=None):
    f = raw.astype('float32')
    f[:,:,:83] = np.arcsinh(f[:,:,:83])
    f[:,:,83:] -= 1
    return pack(f,raw[:,:,0]-.00136,targets,targets is not None)


def external_data():
    content = ARCHIVE.read_bytes()
    oid = hashlib.sha1(f'blob {len(content)}\0'.encode()+content).hexdigest()
    assert oid=='2d8d7749caf622dd07e0df954413dc698129c766'
    manifest = dict(source=SOURCE,sha256=hashlib.sha256(content).hexdigest(),sequences=[])
    del content
    streams = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for fold in (7,8,9):
            name = f'Test_Dst_NoAuction_DecPre_CF_{fold}.txt'
            with archive.open(name) as stream:
                # The first 40 rows are ten levels of prices and sizes. No supplied labels are read.
                a = np.stack([np.fromstring(stream.readline(),sep=' ') for _ in range(40)],axis=1)
            mid = .5*(a[:,0]+a[:,2])
            cuts = np.flatnonzero(np.abs(np.diff(np.log(mid)))>.08)+1
            assert len(cuts)==4
            for idx in np.split(np.arange(len(mid)),cuts):
                b,m = a[idx],mid[idx]
                price = b[:,::2]
                volume = b[:,1::2]
                x = np.column_stack((np.arcsinh(1e4*(price/m[:,None]-1)),
                                     np.log1p(500*volume))).astype('float32')[:,None,:]
                r = 5000*np.diff(np.log(m),prepend=np.log(m[0]))
                streams.append(pack(x,r[:,None]))
                manifest['sequences'].append(dict(file=name,first=int(idx[0]),rows=len(idx)))
    (OUT/'external_data.json').write_text(json.dumps(manifest,indent=2))
    return streams


class BookModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.competition = nn.Linear(100,21)
        self.external = nn.Linear(40,21)
        self.gru = nn.GRU(24,32,batch_first=True)
        self.norm = nn.LayerNorm(32)
        self.proxy = nn.Linear(32,6)
        self.local = nn.Sequential(nn.Linear(64,40),nn.SiLU(),nn.Linear(40,3))
        self.market = nn.Sequential(nn.Linear(64,40),nn.SiLU(),nn.Linear(40,3))

    def encode(self,x,external):
        z = x.float()
        projection = self.external if external else self.competition
        h = torch.cat((F.silu(projection(z[:,:,:-3])),z[:,:,-3:]),2)
        _,state = self.gru(h)
        return self.norm(state[0])

    def forward(self,x):
        h = self.encode(x.reshape(-1,8,x.shape[-1]),False).reshape(len(x),50,32)
        m = h.mean(1,keepdim=True)
        local = self.local(torch.cat((h,m.expand_as(h)),2))
        common = self.market(torch.cat((m[:,0],h.std(1,unbiased=False)),1))
        return local-local.mean(1,keepdim=True)+common[:,None]


def sequences(day,t,stocks=None):
    idx = (t[:,None]-torch.as_tensor(LAGS,device=t.device)).clamp_min(0)
    if stocks is None:
        return day['x'][idx].transpose(1,2)
    return day['x'][idx,stocks[:,None]]


def pretrain(model,days,external,seed):
    rng = np.random.default_rng(seed)
    gen = torch.Generator(device='cuda').manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    for _ in range(PRE_STEPS):
        d = days[int(rng.integers(len(days)))]
        t = torch.randint(64,len(d['proxy']),(192,),generator=gen,device='cuda')
        stocks = torch.randint(d['x'].shape[1],(len(t),),generator=gen,device='cuda')
        z = model.proxy(model.encode(sequences(d,t,stocks),external))
        loss = F.smooth_l1_loss(z,d['proxy'][t,stocks],beta=.55)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()


def fine_tune(model,days):
    rng = np.random.default_rng(SEED+3)
    gen = torch.Generator(device='cuda').manual_seed(SEED+3)
    opt = torch.optim.AdamW(model.parameters(),lr=.0009,weight_decay=.025)
    avg = copy.deepcopy(model).eval()
    for step in range(FIT_STEPS):
        day = days[int(rng.integers(len(days)))]
        t = torch.randint(len(day['y']),(24,),generator=gen,device='cuda')
        y = day['y'][t]
        err = model(sequences(day,t))-y
        m = err.mean(1)
        loss = F.smooth_l1_loss(err-m[:,None],torch.zeros_like(err),beta=1.2)
        loss += .7*F.smooth_l1_loss(m,torch.zeros_like(m),beta=.60)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        with torch.no_grad():
            torch._foreach_lerp_(list(avg.parameters()),list(model.parameters()),.02)
        if step==400:
            for group in opt.param_groups:
                group['lr'] *= .4
    avg.gru.flatten_parameters()
    return avg


def train(end,external,arms):
    days = []
    for d in range(end):
        raw = peer.PARENT.before.book.rawday(d)
        y = peer.labels(d).reshape(-1,50,3)
        days.append(competition(raw,y))
    models = []
    for arm in arms:
        torch.manual_seed(SEED)
        model = BookModel().cuda()
        if arm!='random':
            pretrain(model,external if arm=='external' else days,arm=='external',SEED+1)
            pretrain(model,days,False,SEED+2)
        models.append(fine_tune(model,days))
        print('fit',end,arm,flush=True)
    return models


@torch.inference_mode()
def infer(model,day,t):
    output = []
    for start in range(0,len(t),32):
        ids = torch.from_numpy(t[start:start+32]).cuda()
        output.append(model(sequences(day,ids)).cpu().numpy())
    return np.concatenate(output)


def ic(x,y):
    x = x.reshape(-1,3).astype(float)
    y = y.reshape(-1,3).astype(float)
    x -= x.mean(0)
    y -= y.mean(0)
    return ((x*y).sum(0)/np.sqrt((x*x).sum(0)*(y*y).sum(0))).tolist()


def changes(a,b):
    diff = .15*(a-b)
    m = diff.mean(1,keepdims=True)
    return diff-m,m


def summary(records):
    y,p,random,internal,external = (np.concatenate(v) for v in zip(*records))
    anchor = np.array(ic(p,y))
    gains = [np.array(ic(p+d,y))-anchor for d in changes(external,internal)]
    return dict(baseline=anchor.tolist(),individual=[ic(q,y) for q in (random,internal,external)],
                gains=np.asarray(gains).tolist(),rows=int(y.shape[0]*50))


def screen(streams):
    daily,folds = [],[]
    for end in (8,10,12,14):
        models = train(end,streams,ARMS)
        records = []
        for d in (end,end+1):
            raw = peer.PARENT.before.book.rawday(d)
            t = np.arange(d%4,len(raw),4)
            day = competition(raw)
            preds = [infer(m,day,t) for m in models]
            if d==end:
                n = 197
                short = competition(raw[:n])
                for model,v in zip(models,preds):
                    test = infer(model,short,t[t<n])
                    np.testing.assert_allclose(test,v[:len(test)],rtol=0,atol=5e-5)
            y = peer.labels(d).reshape(-1,50,3)[t]
            records.append((y,peer.reference(d)[t],*preds))
            daily.append(summary(records[-1:]))
        fold = summary(records)
        folds.append(fold)
        print('fold',end+1,'A/B/C',fold['individual'],'delta local/market',fold['gains'],flush=True)
        del models
    gains = np.array([f['gains'] for f in folds])
    keep = (gains.min(0)>0)&((np.array([f['gains'] for f in daily])>0).sum(0)>=5)
    report = dict(arms=ARMS,pretrain_steps=[0,2*PRE_STEPS,2*PRE_STEPS],fit_steps=FIT_STEPS,
                  seed=SEED,folds=folds,daily=daily,retained=keep.tolist())
    (OUT/'results.json').write_text(json.dumps(report,indent=2))
    if keep.any():
        for arm,model in zip(ARMS[1:],train(16,streams,ARMS[1:])):
            torch.save(model.cpu().state_dict(),OUT/f'{arm}.pth')
        np.save(OUT/'mask.npy',keep)
    return keep.any()


def load():
    parent = peer.PARENT.load()
    keep = np.load(OUT/'mask.npy')
    models = []
    for arm in ARMS[1:]:
        m = BookModel().cuda().eval()
        m.load_state_dict(torch.load(OUT/f'{arm}.pth',weights_only=True,map_location='cpu'))
        models.append(m)
    def predict(raw,upstream):
        p = parent(raw,upstream).copy()
        day = competition(raw)
        t = np.arange(len(raw))
        a,b = (infer(model,day,t) for model in models)
        for z,k in zip(changes(b,a),keep):
            p += z*k
        return p
    return predict


def main():
    torch.set_num_threads(2)
    ext = external_data()
    if not screen(ext):
        return
    best = json.loads((peer.PARENT.OUT/'reload_verification.json').read_text())['pooled']['score']
    first = None
    for name in ('raw_replay','reload_verification'):
        result = peer.walk.replay(load())
        if first is not None:
            np.testing.assert_allclose(result['pooled']['candidate'],first['pooled']['candidate'],rtol=0,atol=1e-10)
            for d in result['daily']:
                assert abs(result['daily'][d]['score']-first['daily'][d]['score'])<1e-10
        (OUT/f'{name}.json').write_text(json.dumps(result,indent=2))
        print(name,result['pooled']['score'],flush=True)
        if result['pooled']['score']<=best:
            return
        first = result


if __name__=='__main__':
    main()
