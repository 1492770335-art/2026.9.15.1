import torch
from torch import nn
from torch.nn import functional as F
from .base import BaseModel
class GRU(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        hidden = config['hidden']
        self.gru = nn.GRU(100, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 1)
        self.window = config.get('window', 100)
        self.history = None
    def forward(self, x):
        h, _ = self.gru(x)
        return self.output(h[:, -1])
    def reset(self):
        self.history = None
    def predict(self, x):
        current = x[:, None].detach()
        self.history = current if self.history is None else torch.cat([self.history, current], dim=1)
        self.history = self.history[:, -self.window:]
        history = F.pad(self.history, (0, 0, self.window - self.history.shape[1], 0))
        return self.forward(history)
