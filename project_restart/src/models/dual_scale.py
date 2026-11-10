# -*- coding: utf-8 -*-
"""TC-45 M2-0: dual-time-scale y2 reference architecture (zero-training audit).
- short branch: K16-GRU128 over [t-15,t] x 100 raw features.
- long branch: earlier 48 same-day same-stock ticks [t-63,t-16], f0-f82 (83) only, split into
  6 non-overlapping 8-tick groups (old->new), group mean over VALID ticks (denominator=valid count,
  not 8); empty groups masked (excluded from GRU16 sequence); all-empty -> long rep = zeros(16).
- head: Linear(144,1) on concat[h128, long16].
"""
import torch
from torch import nn
from torch.nn.utils import rnn as rnn_utils


def long_groups_from_window(window64, n):
    """window64: (B,64,100) left-zero-padded, position p = tick(n-63+p); n: (B,) tick_in_day (0-based).
    Returns (groups (B,6,83), vc (B,6) long, lengths (B,) long). Empty groups are zero-filled in
    `groups` but EXCLUDED from the GRU16 sequence via lengths (masked)."""
    B = window64.shape[0]
    long = window64[:, :48, :83]                       # (B,48,83) ticks [t-63 .. t-16]
    valid_start = (63 - n).clamp(min=0, max=48)        # positions >= valid_start are valid
    groups = torch.zeros(B, 6, 83, dtype=window64.dtype, device=window64.device)
    vc = torch.zeros(B, 6, dtype=torch.long, device=window64.device)
    for b in range(B):
        vs = valid_start[b].item()
        for g in range(6):
            lo, hi = 8 * g, 8 * g + 7
            s, e = max(lo, vs), min(hi, 47)
            if s <= e:
                vc[b, g] = e - s + 1
                groups[b, g] = long[b, s:e + 1].mean(0)
    lengths = (vc > 0).sum(1)                          # number of non-empty groups (suffix)
    return groups, vc, lengths


class DualScaleGRU(nn.Module):
    def __init__(self, short_hidden=128, long_hidden=16):
        super().__init__()
        self.short_gru = nn.GRU(100, short_hidden, batch_first=True)
        self.long_gru = nn.GRU(83, long_hidden, batch_first=True)
        self.head = nn.Linear(short_hidden + long_hidden, 1, bias=True)

    def forward(self, window64, n):
        """window64 (B,64,100); n (B,) tick_in_day. Vectorized long branch via pack_padded_sequence."""
        short = window64[:, 48:64, :]
        h, _ = self.short_gru(short)
        hs = h[:, -1]                                  # (B,128)
        groups, vc, lengths = long_groups_from_window(window64, n)
        B = window64.shape[0]
        hl = torch.zeros(B, self.long_gru.hidden_size, dtype=window64.dtype, device=window64.device)
        mask = lengths > 0
        if mask.any():
            idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
            Ls = lengths[idx]
            Ls_sorted, order = torch.sort(Ls, descending=True)
            seqs = [groups[b.item(), (6 - Ls[j].item()):6] for j, b in enumerate(idx)]
            seqs_ordered = [seqs[order[i].item()] for i in range(len(order))]
            packed = rnn_utils.pack_sequence(seqs_ordered, enforce_sorted=True)
            _, hlast = self.long_gru(packed)
            for j in range(len(order)):
                hl[idx[order[j].item()]] = hlast[0, j]
        return self.head(torch.cat([hs, hl], dim=1))


def param_count():
    m = DualScaleGRU()
    return {
        "short_gru": sum(p.numel() for p in m.short_gru.parameters()),
        "long_gru": sum(p.numel() for p in m.long_gru.parameters()),
        "head": sum(p.numel() for p in m.head.parameters()),
        "total": sum(p.numel() for p in m.parameters()),
    }
