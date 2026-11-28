"""Compare endpoint regression to auxiliary regression on graph-decomposed future five-tick rewards."""
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_dense_transition_cash_decomposition_v1 as teacher

strip,rel,onset,REFERENCE,read_p=teacher.strip,teacher.rel,teacher.onset,teacher.REFERENCE,teacher.read_p
OUT=REFERENCE.parent/'dense_transition_cash_student_v1'
ARMS=('path_supervision','above_endpoint_control','above_shuffled_components')
LAGS=(0,5,20,60)
J,S=teacher.J,teacher.S
DEVICE='cuda' if torch.cuda.is_available() else 'cpu'


def history(s,t):
    return np.concatenate([s[(t-h).clip(0)] for h in LAGS],axis=2)


def collect():
    OUT.mkdir(exist_ok=True)
    rule=teacher.cuts()
    np.save(OUT/'cuts.npy',rule)
    edges=teacher.laws(teacher.estimate(rule,range(4)),teacher.estimate(rule,range(4)),True)
    for d in range(16):
        raw,grid=[onset.settled.load(d,k) for k in ('x','time')]
        cols=[]
        for a,b,s,price,gaps in teacher.observations(raw,grid,rule):
            t=np.arange(60+d%5,b-a-60,5)
            ok=gaps[t+61]==gaps[t-60]
            x=history(s,t)
            parts=[]
            for start,stop in ((5,20),(20,60)):
                z=np.zeros((*ok.shape,3))
                for k in range(start,stop,5):
                    for j in range(J):
                        z+=edges[j,s[t+k,:,j],s[t+k+5,:,j]]/J
                actual=price[t+stop]-price[t+start]
                z=np.concatenate((z,(actual-z.sum(-1))[...,None]),axis=-1)
                z=strip.center(z)
                parts.append(z)
            cols.append((x.astype(np.uint8),np.stack(parts,2).astype(np.float32),ok,a+t))
        x,y,ok,t=[np.concatenate(v) for v in zip(*cols)]
        np.savez(OUT/f'day_{d:02d}.npz',x=x,y=y,ok=ok,t=t)


class MarketState(nn.Module):
    def __init__(self):
        super().__init__()
        self.table=nn.Embedding(J*S,6)
        self.frame=nn.Sequential(nn.Linear(J*6,32),nn.LayerNorm(32),nn.SiLU())
        self.summary=nn.Sequential(nn.Linear(len(LAGS)*32,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,24),nn.SiLU())
        self.output=nn.Linear(24,8)

    def forward(self,x):
        ids=x.long().reshape(-1,len(LAGS),J)+torch.arange(J,device=x.device)*S
        state=self.frame(self.table(ids).flatten(-2)).flatten(1)
        return self.output(self.summary(state)).reshape(-1,2,4)


def train_model(data,fit_days):
    x=np.concatenate([data[d][0][data[d][2]] for d in fit_days])
    y=np.concatenate([data[d][1][data[d][2]] for d in fit_days])
    x,y=[torch.as_tensor(v,device=DEVICE) for v in (x,y)]
    scale=y.square().mean(0).sqrt().clamp_min(.008)
    full=y.sum(-1).square().mean(0).sqrt().clamp_min(.03)
    rng=torch.Generator(device=DEVICE).manual_seed(33251)
    ix=torch.randint(len(y),(300,768),device=DEVICE,generator=rng)
    rot=torch.randint(3,(300,768,1,1),device=DEVICE,generator=rng)
    params=[]
    for mode in range(3):
        torch.manual_seed(8544)
        net=MarketState().to(DEVICE)
        optimizer=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.05)
        for i,ids in enumerate(ix):
            pred=net(x[ids])
            target=y[ids]
            loss=F.smooth_l1_loss((pred*scale).sum(-1)/full,target.sum(-1)/full,beta=.75)
            if mode:
                pieces=target/scale
                if mode==2:
                    # Keep the residual return and rotate only the three structured targets.
                    # Rotation is applied in cash units before per-component normalization.
                    order=(torch.arange(3,device=DEVICE)+rot[i]).expand(-1,2,-1)%3
                    pieces=torch.cat((target[...,:3].gather(-1,order),target[...,3:]),-1)/scale
                loss=loss+.35*F.smooth_l1_loss(pred,pieces,beta=.75)
            if i==205:
                optimizer.param_groups[0]['lr']*=.4
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(),1.)
            optimizer.step()
        params.append(nn.utils.parameters_to_vector(net.parameters()).detach().cpu().numpy())
    return np.r_[scale.cpu().numpy().ravel(),np.concatenate(params)]


@torch.inference_mode()
def unpack(bank):
    scale=torch.tensor(bank[:8],device=DEVICE,dtype=torch.float32).reshape(2,4)
    nets=[]
    for packed in np.split(bank[8:],3):
        net=MarketState().to(DEVICE).eval()
        nn.utils.vector_to_parameters(torch.tensor(packed,device=DEVICE,dtype=torch.float32),net.parameters())
        nets.append(net)
    return scale,nets


@torch.inference_mode()
def outputs(x,model):
    scale,nets=model
    allv=[]
    for i in range(0,len(x),32):
        ix=np.minimum(i+np.arange(32),len(x)-1)
        obs=torch.as_tensor(x[ix],device=DEVICE)
        raw=[(net(obs)*scale).sum(-1).reshape(32,50,2).cpu().numpy().astype(float) for net in nets]
        base,true,sham=raw
        allv.append(strip.center(np.stack((true,true-base,true-sham),axis=2))[:min(32,len(x)-i)])
    return np.concatenate(allv)


def readout(data,validation,model):
    stats=[]
    for d in validation:
        if d<4:
            continue
        x,_,_,t=data[d]
        p=strip.center(read_p(d)[t].astype(float))
        y=strip.center(onset.settled.load(d,'y')[t].astype(float))
        z=outputs(x,model)
        mats=[]
        for j in range(3):
            row=np.concatenate((p,z[:,:,j],y[...,1:]),axis=-1).reshape(-1,7)
            mats.append(row.T@row/len(row))
        stats.append(mats)
    stats=np.stack(stats)
    out=np.empty((3,2,5))
    for j in range(3):
        rows=stats[:,j];g=rows.mean(0)
        for h in range(2):
            v=np.zeros(5);v[3+h]=1.
            v[:3]=-np.linalg.solve(g[:3,:3]+.01*np.diag(g.diagonal()[:3]),g[:3,3+h])
            beta=rows[:,h+1,h+5]/rows[:,h+1,h+1].clip(.0001)
            pay=(rows[:,:5,h+5]/beta[:,None].clip(.1)-rows[:,:5,h+1])@v
            m=pay.mean()
            slope=m**3/(m*m+pay.var()/len(pay)+1e-10)/(v@g[:5,:5]@v+.00015)
            out[j,h]=v*slope
    return out


def train(end):
    data=[]
    for d in range(end):
        with np.load(OUT/f'day_{d:02d}.npz') as f:
            data.append(tuple(f[k] for k in ('x','y','ok','t')))
    parts=[]
    for parity in (0,1):
        fit=range(parity,end,2)
        hold=range(1-parity,end,2)
        bank=train_model(data,fit)
        co=readout(data,hold,unpack(bank))
        parts.append(np.r_[bank,co.ravel()])
    return np.r_[np.load(OUT/'cuts.npy'),np.concatenate(parts)]


def signals(raw,grid,p,bank):
    rule=bank[:6*J]
    out=np.zeros((*p.shape[:2],3,2))
    for part in np.split(bank[6*J:],2):
        model=unpack(part[:-30])
        co=part[-30:].reshape(3,2,5)
        for a,b in rel.fields.parent.sessions(grid):
            s=teacher.states(raw[a:b],rule)
            z=outputs(history(s,np.arange(b-a)),model)
            q=strip.center(p[a:b].astype(float))
            for j in range(3):
                out[a:b,:,j]+=.5*np.concatenate((q,z[:,:,j]),axis=-1)@co[j].T
    return out


def predict(raw,grid,p,params):
    return strip.evaluation.apply(sys.modules[__name__],raw,grid,p,params)


if __name__=='__main__':
    torch.set_num_threads(2)
    collect()
    strip.evaluation.rolling(sys.modules[__name__])
