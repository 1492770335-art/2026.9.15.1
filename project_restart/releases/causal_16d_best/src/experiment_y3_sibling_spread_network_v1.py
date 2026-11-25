"""Estimate portfolio spreads with a neural head that changes sign when the two portfolios are swapped."""
from functools import lru_cache
import sys

import numpy as np
import torch
from torch import nn

import joint_factor_stage_references as history
import experiment_y3_liquidity_tree_spreads_v1 as graph

trial,study,parent=history.trial,history.study,history.incumbent
OUT=trial.clock.ART/'y3_sibling_spread_network_v1'
ARMS=('independent sides','joint sibling forecast','joint minus independent','moderate return head',
      'moderate and total','wrong pair identities','cross-side interaction alone','fine sibling spreads',
      'wide sibling spreads','later horizons','conditional on spread state')
CANDIDATES=(1,2,3,4,6,7,8,9,10)
CONTROLS=(0,5)


def summarize(raw,t,st,wrong=False):
    z=graph.factors.clocks.signature.impact.arrival.signals(raw)
    fields=np.tanh(.65*np.arcsinh(raw[:,:,[7,11,14,21,38,45,54,55,77,78]].astype('float32')))
    v=np.concatenate((np.tanh(z),fields),2)
    path=np.concatenate((np.zeros_like(v[:1],dtype=float),v.cumsum(0,dtype=float)))
    stop=t+1
    def avg(h):
        beg=np.maximum(stop-h,np.where(t>=2401,2401,0))
        return ((path[stop]-path[beg])/(stop-beg)[:,None,None]).astype('float32')
    a,b,c=avg(4),avg(20),avg(96)
    x=np.concatenate((v[t]-a,a-b,b-c,c),2)
    def side(key):
        w=st[key]
        if wrong:
            w=np.roll(w,17,1)
        return (w@x)*np.sqrt(1/(w*w).sum(1))[None,:,None].clip(1,3)
    return np.stack((side('left'),side('right')),2)


@lru_cache(maxsize=16)
def record(d):
    st=graph.tree(4)
    raw=trial.clock.book.rawday(d)
    t=np.arange(83+d%9,len(raw)-60,9)
    t=t[(t+60<2401)|(t>=2401)]
    x=summarize(raw,t,st)
    y=trial.clock.labels(d).reshape(-1,50,3)
    z=st['nodes']@trial.clock.center(np.stack([y[t+h,:,1] for h in (0,20,40)],2))
    return x.astype('float16'),z.astype('float32')


class PairNet(nn.Module):
    def __init__(self,width,joint):
        super().__init__()
        self.joint=joint
        self.enc=nn.Sequential(nn.Linear(width,48),nn.LayerNorm(48),nn.SiLU(),nn.Linear(48,24))
        self.read=nn.Sequential(nn.Linear(72 if joint else 24,40),nn.LayerNorm(40),nn.SiLU(),nn.Linear(40,6))
    def forward(self,x):
        a,b=self.enc(x).unbind(1)
        if self.joint:
            matched=a*b
            return .5*(self.read(torch.cat((a,b,matched),1))-self.read(torch.cat((b,a,matched),1)))
        return .5*(self.read(a)-self.read(b))


@lru_cache(maxsize=7)
def state(end):
    device='cuda' if torch.cuda.is_available() else 'cpu'
    xs,ys=[np.concatenate(c) for c in zip(*(record(d) for d in range(end)))]
    sigma=np.sqrt(np.mean(ys**2,axis=0)+.10).astype('float32')
    y=ys/sigma
    target=np.concatenate((y.clip(-3,3),y.clip(-1.8,1.8)-y.clip(-.55,.55)),2).reshape(-1,6)
    z=torch.tensor(target,device=device)
    width=xs.shape[-1]
    x=torch.tensor(xs.reshape(-1,2,width),device=device)
    std=x[::13].float().std((0,1)).clamp(min=.04)
    st=dict(**graph.tree(4),sigma=sigma,std=std.cpu().numpy())
    for joint in (False,True):
        torch.manual_seed(32741)
        net=PairNet(width,joint).to(device)
        opt=torch.optim.AdamW(net.parameters(),lr=.0014,weight_decay=.025)
        for it in range(350):
            ix=torch.randint(len(x),(768,),device=device)
            pred=net((x[ix].float()/std).clamp(-6,6))
            loss=(pred-z[ix]).square().mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),1.5)
            opt.step()
        st.update({f'm{int(joint)}_{k}':v.detach().cpu().numpy() for k,v in net.state_dict().items()})
    return st


def networks(st):
    nets=[]
    for joint in (False,True):
        net=PairNet(len(st['std']),joint)
        net.load_state_dict({k:torch.from_numpy(st[f'm{int(joint)}_{k}']) for k in net.state_dict()})
        nets.append(net.eval())
    return nets


@torch.inference_mode()
def features(raw,t,st,nets):
    right=summarize(raw,t,st)
    wrong=summarize(raw,t,st,True)
    parts=[]
    for i in range(0,len(t),64):
        def flatten(x):
            a=torch.from_numpy((x[i:i+64]/st['std']).clip(-6,6))
            return a.flatten(0,1),len(a)
        x,n=flatten(right)
        bad,_=flatten(wrong)
        a,b=[net(x).numpy().reshape(n,49,6) for net in nets]
        fake=nets[1](bad).numpy().reshape(n,49,6)[:,:,:3]
        # Compare the interaction with marginal effects under the very same joint network.
        one=x.clone();one[:,1]=0
        two=x.clone();two[:,0]=0
        interact=b[:,:,:3]-(nets[1](one)+nets[1](two)).numpy().reshape(n,49,6)[:,:,:3]
        delta=b[:,:,:3]-a[:,:,:3]
        def lift(y,h):
            return h.T@(y*st['sigma'])
        h=st['nodes']
        cols=[lift(y,h) for y in (a[:,:,:3],b[:,:,:3],delta,b[:,:,3:])]
        cols.extend((lift(fake,np.roll(h,17,axis=1)),lift(interact,h),
                     lift(delta*(st['levels']==0)[None,:,None],h),lift(delta*(st['levels']==2)[None,:,None],h),
                     lift(interact*np.tanh(abs(b[:,:,:3])*3),h)))
        parts.append(np.concatenate(cols,2))
    return trial.clock.center(np.concatenate(parts))


def historical_design(d):
    st=state(max(4,d-d%2)); nets=networks(st);cache={}
    def build(raw,p,t):
        key=len(raw),t.tobytes()
        if key not in cache:
            cache[key]=features(raw,t,st,nets)
        return np.concatenate((trial.clock.center(p[t]),cache[key]),2)
    return build


def fit(rows):
    s=[np.arange(3+3*i,6+3*i) for i in range(9)]
    groups=[*s[:4],np.r_[s[1],s[3]],*s[4:8],s[1][1:],s[8]]
    return dict(**graph.factors.clocks.linear.gram(rows,[(g,3,.38) for g in groups]),**state(len(rows)+4))


changes=graph.changes


def correction():
    with np.load(OUT/'model.npz') as f: st=dict(f)
    nets=networks(st)
    def apply(raw,p):
        z=features(raw,np.arange(len(raw)),st,nets)
        return changes(np.concatenate((trial.clock.center(p),z),2),st)[int(st['pick'])]
    return apply


def layer():
    return study.layer(sys.modules[__name__],parent)


if __name__=='__main__':
    torch.set_num_threads(2)
    study.score_run(sys.modules[__name__],reference=history)
