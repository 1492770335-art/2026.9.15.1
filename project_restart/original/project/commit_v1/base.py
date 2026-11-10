from abc import ABC, abstractmethod
from torch import nn


class BaseModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def predict(self, x):
        pass
