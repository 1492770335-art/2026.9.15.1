"""Paired independent, shared, and shared/private GRUs for targets y1/y2."""
import copy
import torch
from torch import nn

class JointMarketGRU(nn.Module):
    def __init__(self, variant='shared_private'):
        super().__init__()
        if variant not in ('independent','shared','shared_private'):
            raise ValueError(variant)
        self.variant=variant
        self.shared=nn.GRU(300,128,batch_first=True)
        if variant=='shared_private':
            self.private1=nn.GRU(300,64,batch_first=True)
            self.private2=nn.GRU(300,64,batch_first=True)
            self.head1=nn.Linear(192,1)
            self.head2=nn.Linear(192,1)
        else:
            self.head1=nn.Linear(128,1)
            self.head2=copy.deepcopy(self.head1)
            if variant=='independent':self.independent2=copy.deepcopy(self.shared)

    def forward(self,x):
        h,_=self.shared(x);h=h[:,-1]
        if self.variant=='independent':
            h2,_=self.independent2(x);h2=h2[:,-1]
            return torch.cat([self.head1(h),self.head2(h2)],-1)
        if self.variant=='shared_private':
            a,_=self.private1(x);b,_=self.private2(x)
            return torch.cat([self.head1(torch.cat([h,a[:,-1]],-1)),
                              self.head2(torch.cat([h,b[:,-1]],-1))],-1)
        return torch.cat([self.head1(h),self.head2(h)],-1)
