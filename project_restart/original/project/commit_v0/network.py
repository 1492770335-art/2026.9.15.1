from torch import nn
from .base import BaseModel


class MLP(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        hidden = config['hidden']
        self.network = nn.Sequential(nn.Linear(100, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.network(x)

    def reset(self):
        pass

    def predict(self, x):
        return self.forward(x)
