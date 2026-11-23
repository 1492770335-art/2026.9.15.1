"""Estimate an odd price-pressure response with liquidity-dependent magnitudes and cross-stock factors."""
import json
import numpy as np
import torch
from torch import nn
import experiment_phase_future_book_v1 as phase
import experiment_variance_ratio_calibration_v1 as benchmark

OUT=phase.inn.prior.ART/'symmetric_pressure_network_v1'
GROUPS=((20,21,22,23),(24,25,26,27),(28,29,30,31),(60,61),(64,65),(68,69),(72,73))
PAIRS=((41,42),(50,51),(52,53),(54,55),(56,57))
D=22
N_SIGN=5*D


def signed(raw):
    def field(j):return np.arcsinh(raw[:,:,j].astype(np.float32))
    separate=[field(k) for k in (0,10,11,13,14,15,34,36,38,74)]
    groups=[sum(field(j) for j in g)/len(g) for g in GROUPS]
    differences=[field(i)-field(j) for i,j in PAIRS]
    return np.stack(separate+groups+differences,axis=2)


def features(raw,t):
    s=signed(raw)
    c=np.concatenate((np.zeros_like(s[:1],dtype=float),s.cumsum(0,dtype=float)))
    parts=[s[t]];ix=t+1
    for a,b in ((1,5),(5,20),(20,70),(70,230)):
        hi=(ix-a).clip(0);lo=(ix-b).clip(0)
        parts.append(((c[hi]-c[lo])/(hi-lo).clip(1)[:,None,None]).astype(np.float32))
    r=raw[:,:,0].astype(float)-.00136
    vol=np.sqrt((8+phase.inn.moving_sum(r*r,96))/(8+np.minimum(np.arange(len(r))+1,96))[:,None]).clip(.45,4.)
    u=[np.log(.15+vol[t]),(phase.inn.moving_sum(abs(r)>.018,64)/np.minimum(np.arange(len(r))+1,64)[:,None])[t]]
    for i,j in PAIRS:u.append(np.arcsinh(abs(raw[t,:,i].astype(float))+abs(raw[t,:,j].astype(float))))
    for j in (7,8,9,16,37,39,40,77,78,79,80,81,82):
        u.append(np.arcsinh(raw[t,:,j].astype(float)))
    u.append(np.exp(-t[:,None]/350)*np.ones((1,50)))
    parts.append(np.stack(u,axis=2))
    return np.concatenate(parts,axis=2).clip(-8,8).astype('float16'),vol[t].astype(np.float32)


def auxiliary(raw,t):
    s=signed(raw)
    c=np.concatenate((np.zeros_like(s[:1],dtype=float),s.cumsum(0,dtype=float)))
    return np.concatenate([((c[t+b]-c[t+a])/(b-a)-s[t]).clip(-4,4)
                           for a,b in ((1,6),(6,21),(21,61))],axis=2).astype(np.float16)


class PressureNet(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.sign=nn.Linear(N_SIGN,96,bias=False)
        self.gate=nn.Sequential(nn.Linear(n,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.1),nn.Linear(96,96))
        self.identity=nn.Parameter(torch.zeros(50,96))
        self.factor=nn.Linear(96,24,bias=False)
        self.local=nn.Linear(144,96,bias=False)
        self.lgate=nn.Sequential(nn.Linear(144,96),nn.LayerNorm(96),nn.SiLU(),nn.Dropout(.14),nn.Linear(96,96))
        self.r=nn.Linear(96,6,bias=False)
        self.shared=nn.Linear(120,64,bias=False)
        self.sgate=nn.Sequential(nn.Linear(144,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.18),nn.Linear(64,64))
        self.m=nn.Linear(64,6,bias=False);self.future=nn.Linear(96,3*D,bias=False)
        self.register_buffer('port',torch.zeros(4,50));self.register_buffer('beta',torch.ones(50))
    def forward(self,x,v,opening):
        s=x[:,:,:N_SIGN]
        even=torch.cat((s.abs(),x[:,:,N_SIGN:]),2)
        h=torch.tanh(self.sign(s))*torch.sigmoid(self.gate(even)+self.identity)*2
        z=self.factor(h);f=self.port@z;m=z.mean(1,keepdim=True)
        raw=torch.cat((h,m.expand_as(z),self.port.T@f),2)
        loc=torch.tanh(self.local(raw))*torch.sigmoid(self.lgate(abs(raw)))*2
        a,b=self.r(loc).chunk(2,-1);r=(a+b*opening[:,None])*v[:,:,None];r=r-r.mean(1,keepdim=True)
        direct=torch.cat((m[:,0],f.flatten(1)),1)
        shared=torch.tanh(self.shared(direct))*torch.sigmoid(self.sgate(torch.cat((direct.abs(),z.std(1,unbiased=False)),1)))*2
        a,b=self.m(shared).chunk(2,-1);market=(a+b*opening)*v.mean(1,keepdim=True)
        return r+market[:,None]*self.beta[None,:,None],self.future(loc),r,market


def main():
    OUT.mkdir(exist_ok=True);hist=[];report={}
    for end,stop in ((12,14),(14,16),(16,20)):
        ck=OUT/f'net_{end}.pth'
        if ck.exists():
            state=torch.load(ck,map_location='cpu',weights_only=True)
            model=PressureNet(state['gate.0.weight'].shape[1]).cuda().eval();model.load_state_dict(state)
        else:model=phase.train(end,seed=9883,out=OUT,model_class=PressureNet,make_features=features,make_aux=auxiliary)
        rows=[]
        for d in range(end,stop):
            raw=phase.inn.grid(d);n=phase.infer(model,raw,make_features=features)
            if d==end:np.testing.assert_allclose(n[:153],phase.infer(model,raw[:153],make_features=features),atol=1e-4)
            p=benchmark.baseline(d);y=phase.inn.prior.labels(d).reshape(p.shape)
            np.save(OUT/f'side_{d}.npy',n);rows.append((p,y,n))
        if end==12:_,w=phase.fits.coefficients(rows)
        entries=[]
        for d,(p,y,n) in zip(range(end,stop),rows):
            q=phase.v1.blend(p,n,w).astype(np.float32);np.save(OUT/f'pred_{d}.npy',q)
            entries.append(tuple(a.reshape(-1,3) for a in (p,q,y)))
        result=phase.inn.prior.measure(entries);report[str(stop)]={'w':w.tolist(),'result':result}
        (OUT/'results.json').write_text(json.dumps(report,indent=2));print(stop,report[str(stop)],flush=True)
        if end<16:hist.extend(rows)
        if end==14:
            if result['gain']<=0:break
            _,w=phase.fits.coefficients(hist)

if __name__=='__main__':main()
