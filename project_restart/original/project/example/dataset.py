import numpy as np
import polars as pl
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset


class ExampleDataset(Dataset):
    def __init__(self, frame: pl.DataFrame, window, device, label):
        self.window = window
        features = sorted([name for name in frame.columns if name.startswith('f')], key=lambda name: int(name[1:]))
        x = frame.select(features).to_numpy().astype(np.float16)
        y = frame.select(f'y{label}').to_numpy().astype(np.float16)
        self.x = torch.from_numpy(x).to(device)
        self.y = torch.from_numpy(y).to(device)
        dates = frame['date'].to_numpy()
        changed = np.r_[True, dates[1:] != dates[:-1]]
        self.starts = np.maximum.accumulate(np.where(changed, np.arange(len(frame)), 0))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        if self.window is None:
            return self.x[index], self.y[index]
        length = min(self.window, (index - self.starts[index]) // 50 + 1)
        x = self.x[index - (length - 1) * 50:index + 1:50]
        return F.pad(x, (0, 0, self.window - length, 0)), self.y[index]
