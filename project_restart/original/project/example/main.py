import argparse
import json
import os
import polars as pl
import torch

from dataset import ExampleDataset
from network import MLP, GRU
from fit import fit_sft

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)
    
    torch.set_num_threads(config['num_threads'])
    torch.manual_seed(config['seed'])
    df = pl.read_parquet(config['data'])
    dataset = ExampleDataset(df, config['window'], config['device'], config['label'])
    output = os.path.join(config['output'], config['model'], config['name'])
    result = os.path.join(output, 'result')
    os.makedirs(result, exist_ok=True)
    if config['model'] == 'MLP':
        model = MLP(config)
    elif config['model'] == 'GRU':
        model = GRU(config)
    model = model.to(config['device'])
    fit_sft(model, dataset, config, output)

if __name__ == '__main__':
    main()
