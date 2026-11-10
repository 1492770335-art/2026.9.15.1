from torch import nn
import torch
from .base import BaseModel

class Ridge(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.W = nn.Parameter(torch.zeros(101, 1))

    def reset(self):
        pass

    def predict(self, x):
        ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        return torch.cat([x, ones], dim=1) @ self.W
