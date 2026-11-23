import argparse
import importlib.util
import json
import os
import sys
from tqdm import tqdm

import numpy as np
import polars as pl
import torch


class Evaluator:
    def __init__(self, parquet_path, commit_path, device='cpu'):
        self.parquet_path = parquet_path
        self.device = device
        module_name = f'_submission_{id(self)}'
        spec = importlib.util.spec_from_file_location(module_name, os.path.join(commit_path, 'network.py'), submodule_search_locations=[os.path.abspath(commit_path)])
        network = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = network
        spec.loader.exec_module(network)
        self.models = []
        for label in range(3):
            with open(os.path.join(commit_path, 'conf', f'Model_{label}.json')) as f:
                config = json.load(f)
            model = getattr(network, config['model'])(config).to(device)
            path = os.path.join(commit_path, 'model', f'Model_{label}.pth')
            model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            model.eval()
            self.models.append(model)

    @torch.no_grad()
    def evaluate(self, date=None):
        data = pl.scan_parquet(self.parquet_path)
        if date is None:
            dates = data.select(pl.col('date').unique().sort()).collect()['date'].to_list()
        else:
            dates = [date] if isinstance(date, int) else sorted(set(date))
        sums = np.zeros((6, 3), dtype=np.float64)
        for day in tqdm(dates, desc= 'Evaluating...'):
            frame = data.filter(pl.col('date') == day).sort(['time', 'code']).collect()
            if frame.is_empty():
                raise ValueError(f'date={day} 不存在')
            features = sorted([name for name in frame.columns if name.startswith('f')], key=lambda name: int(name[1:]))
            x = frame.select(features).to_numpy().astype(np.float32).reshape(-1, 50, 100)
            x = torch.from_numpy(x).to(self.device)
            y = frame.select(['y0', 'y1', 'y2']).to_numpy().astype(np.float64)
            predictions = torch.empty((len(x), 50, 3), device=self.device)
            for model in self.models:
                model.reset()
            for tick, current in enumerate(x):
                for label, model in enumerate(self.models):
                    prediction = model.predict(current.clone())
                    if prediction.shape != (50, 1):
                        raise ValueError(f'model{label}: date={day}, tick={tick}, shape={tuple(prediction.shape)}, expected=(50, 1)')
                    predictions[tick, :, label] = prediction[:, 0]
            if not torch.isfinite(predictions).all():
                tick, code, label = (~torch.isfinite(predictions)).nonzero()[0].tolist()
                raise ValueError(f'model{label}: date={day}, tick={tick}, code={code}, prediction is NaN/Inf')
            p = predictions.cpu().numpy().astype(np.float64).reshape(-1, 3)
            sums += np.stack((np.full(3, len(y)), p.sum(0), y.sum(0), (p * p).sum(0), (y * y).sum(0), (p * y).sum(0)))
        n, p, y, pp, yy, py = sums
        denominator = np.sqrt(np.maximum(pp - p * p / n, 0) * np.maximum(yy - y * y / n, 0))
        if (denominator == 0).any():
            raise ValueError('invalid predict')
        corr = (py - p * y / n) / denominator
        return dict(corr0=round(float(corr[0]), 4), corr1=round(float(corr[1]), 4), corr2=round(float(corr[2]), 4), score=round(float(corr @ np.array([0.2, 0.3, 0.5])), 4))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True)
    parser.add_argument('--commit_dir', default='commit')
    parser.add_argument('--date', type=int, nargs='+')
    parser.add_argument('--device', default='cuda:0') # 'cpu'
    args = parser.parse_args()
    evaluator = Evaluator(args.data, args.commit_dir, args.device)
    print(json.dumps(evaluator.evaluate(args.date), indent=2))
