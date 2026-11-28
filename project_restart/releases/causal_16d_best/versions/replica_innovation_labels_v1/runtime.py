"""Shared configuration, chronological dependencies and deterministic CPU thread count."""
import argparse
import importlib
import os
from pathlib import Path
import sys

from layers import LAYERS

HERE = Path(__file__).resolve().parent
PREVIOUS = HERE.parent / 'basket_cash_feedback_v1'


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--run-dir', required=True, type=Path)
    p.add_argument('--prior-src', required=True, type=Path)
    p.add_argument('--rolling', type=Path)
    return p


def configure(args):
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 6000))
    os.environ['TULIN_RUN_DIR'] = str(args.run_dir.resolve())
    os.environ['TULIN_PRIOR_SRC'] = str(args.prior_src.resolve())
    if args.rolling:
        os.environ['TULIN_ROLLING'] = str(args.rolling.resolve())
    sys.path[:0] = [str(HERE / 'src'), str(PREVIOUS / 'src'), str(args.prior_src.resolve())]
    import torch
    torch.set_num_threads(2)
    importlib.import_module('experiment_dense_market_conditional_graph_risk_v1')


def modules():
    return [importlib.import_module(layer[0]) for layer in LAYERS]
