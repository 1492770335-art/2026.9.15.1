"""Represent a stock's feature families as separate tokens with target-specific attention."""
import numpy as np
import torch
from torch import nn
import experiment_phase_future_book_v1 as phase
from experiment_crossfit_predictable_tails_v1 import run_validation

OUT=phase.inn.prior.ART/'target_book_family_tokens_v1'
GROUPS=((0,1,2,3,4,5,6,17,18,19),tuple(range(7,17)),tuple(range(20,24)),
        tuple(range(24,28)),tuple(range(28,34)),tuple(range(34,41)),
        tuple(range(41,50)),tuple(range(50,58)),tuple(range(58,66)),
        tuple(range(66,77)),tuple(range(77,83)),tuple(range(83,100)))


def columns(width):
    # Match the causal current and lag-difference columns of spline.features.
    indices=[]
    fields=phase.spline.FIELDS
    for group in GROUPS:
        cols=list(group)
        for lag in range(3):
            cols.extend(100+lag*len(fields)+i for i,k in enumerate(fields) if k in group)
        indices.append(cols)
    indices.append(list(range(100+3*len(fields),width)))
    assert sorted(sum(indices,[]))==list(range(width))
    return indices


class FamilyNet(nn.Module):
    def __init__(self,width):
        super().__init__()
        groups=columns(width)
        self.project=nn.ModuleList([nn.Linear(len(ids),28) for ids in groups])
        for i,ids in enumerate(groups):
            self.register_buffer(f'idx_{i}',torch.tensor(ids))
        self.family=nn.Parameter(torch.randn(len(groups),28)*.025)
        self.stock=nn.Parameter(torch.randn(50,28)*.015)
        self.family_norm=nn.LayerNorm(28)
        self.context=nn.Sequential(nn.Linear(84,40),nn.SiLU(),nn.Linear(40,28))
        self.key=nn.Sequential(nn.LayerNorm(28),nn.Linear(28,24))
        self.value=nn.Sequential(nn.Linear(28,40),nn.SiLU(),nn.Linear(40,40))
        self.query=nn.Parameter(torch.randn(3,24)*.17)
        self.state_query=nn.Linear(28,24)
        self.stock_attn=nn.MultiheadAttention(40,4,dropout=.10,batch_first=True)
        self.stock_norm=nn.LayerNorm(40)
        self.local=nn.Sequential(nn.Linear(120,80),nn.LayerNorm(80),nn.SiLU(),nn.Dropout(.14),nn.Linear(80,2))
        self.market=nn.Sequential(nn.Linear(240,64),nn.LayerNorm(64),nn.SiLU(),nn.Dropout(.17),nn.Linear(64,2))
        self.future=nn.Sequential(nn.Linear(120,80),nn.SiLU(),nn.Linear(80,66))
        self.register_buffer('port',torch.zeros(4,50))
        self.register_buffer('beta',torch.ones(50))

    def forward(self,x,vol,opening):
        tokens=torch.stack([layer(x.index_select(-1,getattr(self,f'idx_{i}'))) for i,layer in enumerate(self.project)],dim=2)
        tokens=self.family_norm(tokens+self.family+self.stock[:,None])
        b,s,k,h=tokens.shape
        flat=tokens.reshape(b,s,k*h)
        peer=(self.port.T@(self.port@flat)).reshape(b,s,k,h)
        market=tokens.mean(1,keepdim=True).expand_as(tokens)
        tokens=tokens+self.context(torch.cat((tokens,peer,market),axis=-1))
        query=self.query+self.state_query(tokens.mean(2))[:,:,None,:]
        score=query@self.key(tokens).transpose(-1,-2)/np.sqrt(24.)
        if self.training:
            # Use one common field mask per frame so the cross-stock comparison remains well defined.
            missing=torch.rand((b,1,1,k),device=x.device)<.065
            missing[:,:,:,-1]=False
            score=score.masked_fill(missing,-10000.)
        z=score.softmax(-1)@self.value(tokens)
        summary=self.stock_norm(z.mean(2))
        response=self.stock_attn(summary,summary,summary,need_weights=False)[0]
        z=z+response[:,:,None]
        f=(self.port@z.reshape(b,s,-1)).reshape(b,4,3,40)
        peer=(self.port.T@f.reshape(b,4,-1)).reshape(b,s,3,40)
        m=z.mean(1)
        a,k=self.local(torch.cat((z,peer,m[:,None].expand_as(z)),dim=3)).unbind(-1)
        loc=(a+k*opening[:,None,:])*vol[:,:,None]
        loc=loc-loc.mean(1,keepdim=True)
        a,k=self.market(torch.cat((m,z.std(1,unbiased=False),f.transpose(1,2).reshape(b,3,160)),dim=2)).unbind(-1)
        shared=(a+k*opening)*vol.mean(1,keepdim=True)
        return loc+self.beta[None,:,None]*shared[:,None],self.future(z.flatten(2)),loc,shared


def restore(state):
    width=sum(len(state[f'idx_{i}']) for i in range(len(GROUPS)+1))
    return FamilyNet(width)


def main():
    run_validation(OUT,lambda end:phase.train(end,seed=8261,out=OUT,model_class=FamilyNet),restore)


if __name__=='__main__':main()

