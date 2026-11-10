from abc import ABC, abstractmethod
from torch import nn

class BaseModel(nn.Module, ABC):
    def __init__(self, config):
        super().__init__()
        self.config = config

    @abstractmethod
    def reset(self): # 跨日重置
        pass

    @abstractmethod
    def predict(self, x): # x.size() = (50, 100), 每个时间步的所有截面特征
        pass
