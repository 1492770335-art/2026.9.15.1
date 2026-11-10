import torch
from torch import nn
from torch.nn import functional as F
from .base import BaseModel
class _GRU100(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gru = nn.GRU(100, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)
    def forward(self, x):
        h, _ = self.gru(x)
        return self.output(h[:, -1])
class _GRU300(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gru = nn.GRU(300, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)
    def forward(self, x):
        h, _ = self.gru(x)
        return self.output(h[:, -1])
class MarketEnsemble(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        hidden = config['hidden']
        self.window = config.get('window', 16)
        self.members = nn.ModuleList([_GRU300(hidden) for _ in range(3)])
        self.hist = [None, None, None]
    def reset(self):
        self.hist = [None, None, None]
    def predict(self, x):
        m = x.mean(0, keepdim=True)
        s = x.std(0, keepdim=True)
        xm = torch.cat([x, m.expand_as(x), s.expand_as(x)], dim=1)
        cur = xm[:, None].detach()
        outs = []
        for k, mem in enumerate(self.members):
            self.hist[k] = cur if self.hist[k] is None else torch.cat([self.hist[k], cur], dim=1)
            self.hist[k] = self.hist[k][:, -self.window:]
            h = F.pad(self.hist[k], (0, 0, self.window - self.hist[k].shape[1], 0))
            outs.append(mem(h))
        return torch.stack(outs, 0).mean(0)
class K16Ensemble(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        hidden = config['hidden']
        self.window = config.get('window', 16)
        self.members = nn.ModuleList([_GRU100(hidden) for _ in range(3)])
        self.hist = [None, None, None]
    def reset(self):
        self.hist = [None, None, None]
    def predict(self, x):
        cur = x[:, None].detach()
        outs = []
        for k, mem in enumerate(self.members):
            self.hist[k] = cur if self.hist[k] is None else torch.cat([self.hist[k], cur], dim=1)
            self.hist[k] = self.hist[k][:, -self.window:]
            h = F.pad(self.hist[k], (0, 0, self.window - self.hist[k].shape[1], 0))
            outs.append(mem(h))
        return torch.stack(outs, 0).mean(0)
