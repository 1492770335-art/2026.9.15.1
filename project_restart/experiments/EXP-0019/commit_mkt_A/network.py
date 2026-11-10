import torch
from torch import nn
from torch.nn import functional as F
from .base import BaseModel
class GRU(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        hidden = config['hidden']
        self.gru = nn.GRU(300, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)
        self.window = config.get('window', 16)
        self.history = None
    def forward(self, x):
        h, _ = self.gru(x)
        return self.output(h[:, -1])
    def reset(self):
        self.history = None
    def predict(self, x):
        m = x.mean(0, keepdim=True)
        s = x.std(0, keepdim=True)
        xm = torch.cat([x, m.expand_as(x), s.expand_as(x)], dim=1)
        cur = xm[:, None].detach()
        self.history = cur if self.history is None else torch.cat([self.history, cur], dim=1)
        self.history = self.history[:, -self.window:]
        h = F.pad(self.history, (0, 0, self.window - self.history.shape[1], 0))
        return self.forward(h)
