import torch
from torch import nn
from torch.nn import functional as F
from .base import BaseModel
class _GRU(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.gru = nn.GRU(100, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)
    def forward(self, x):
        h, _ = self.gru(x)
        return self.output(h[:, -1])
class GRUEnsemble(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        hidden = config['hidden']
        self.window = config.get('window', 100)
        self.members = nn.ModuleList([_GRU(hidden) for _ in range(3)])
        self.hist = [None, None, None]
    def reset(self):
        self.hist = [None, None, None]
    def predict(self, x):
        outs = []
        for k, m in enumerate(self.members):
            cur = x[:, None].detach()
            self.hist[k] = cur if self.hist[k] is None else torch.cat([self.hist[k], cur], dim=1)
            self.hist[k] = self.hist[k][:, -self.window:]
            h = F.pad(self.hist[k], (0, 0, self.window - self.hist[k].shape[1], 0))
            outs.append(m(h))
        return torch.stack(outs, dim=0).mean(dim=0)
