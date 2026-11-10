# -*- coding: utf-8 -*-
"""TC-42 M0: K16-GRU128 + K4 local depthwise-conv branch (multiscale, pre-registered reference).
Architecture (FROZEN at M0):
  - backbone: GRU(100,128) over K16 window, last hidden h (128).
  - local branch: DepthwiseConv1d(100,100,k=4,groups=100,bias=True) over the same window (channels=features),
    ReLU, take last timestep (summarizes last 4 ticks) -> (100,), Linear(100,32) -> local (32).
  - head: Linear(160,1) on concat[h, local].
Causal: conv kernel=4 no padding, last position uses only ticks [t-3..t]; no future. 100-dim input,
f83-f99 ordinal values preserved (no diff/clip/onehot/MKT).
"""
import torch
from torch import nn
from torch.nn import functional as F


class MultiScaleK4GRU(nn.Module):
    def __init__(self, hidden=128, local_dim=32, window=16, in_dim=100):
        super().__init__()
        self.window = window
        self.gru = nn.GRU(in_dim, hidden, batch_first=True)
        self.local_conv = nn.Conv1d(in_dim, in_dim, kernel_size=4, groups=in_dim, bias=True)  # depthwise
        self.local_proj = nn.Linear(in_dim, local_dim, bias=True)
        self.head = nn.Linear(hidden + local_dim, 1, bias=True)
        self.history = None  # streaming state (set/cleared by caller)

    def forward(self, x):  # x (B, window, 100)
        h, _ = self.gru(x)                    # (B, window, 128)
        gru_h = h[:, -1]                      # (B, 128)
        xt = x.transpose(1, 2)                # (B, 100, window)
        c = self.local_conv(xt)               # (B, 100, window-3)
        c = F.relu(c)
        c = c[:, :, -1]                       # (B, 100) last 4-tick summary
        local = self.local_proj(c)            # (B, 32)
        return self.head(torch.cat([gru_h, local], dim=1))  # (B,1)

    def forward_stream(self, gru_seq, buf4):  # gru_seq (B,T,100) current history; buf4 (B,4,100) recent 4 ticks (zero-padded)
        h, _ = self.gru(gru_seq)
        gru_h = h[:, -1]
        xt = buf4.transpose(1, 2)             # (B,100,4)
        c = self.local_conv(xt)               # (B,100,1)
        c = F.relu(c)[:, :, 0]
        local = self.local_proj(c)
        return self.head(torch.cat([gru_h, local], dim=1))


def param_count():
    m = MultiScaleK4GRU()
    return {
        "gru": sum(p.numel() for p in m.gru.parameters()),
        "local_conv": sum(p.numel() for p in m.local_conv.parameters()),
        "local_proj": sum(p.numel() for p in m.local_proj.parameters()),
        "head": sum(p.numel() for p in m.head.parameters()),
        "total": sum(p.numel() for p in m.parameters()),
    }
