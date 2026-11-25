"""Frozen peer graph and volume calibration for quote-event messages."""
from functools import lru_cache

import numpy as np

import experiment_y3_transaction_pressure_network_v1 as network
import y3_reconstructed_touch_queues as queues

trial, study, history, book = network.trial, network.study, network.history, network.book


def balanced(graph):
    # Column j sends to stock j. Both the diagonal and every column sum stay zero.
    mask = 1 - np.eye(len(graph))
    return graph - mask * graph.sum(0) / (len(graph)-1)


def recent(cumulative, ticks, window):
    return (cumulative[ticks+1] - cumulative[(ticks+1-window).clip(0)]) / np.sqrt(window)


@lru_cache(maxsize=7)
def params(end):
    return dict(queues.calibration(), graphs=network.params(end)['graphs'])
