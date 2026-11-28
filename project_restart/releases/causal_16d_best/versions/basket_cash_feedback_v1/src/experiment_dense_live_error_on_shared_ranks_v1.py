"""Test the previously selected live-memory innovation against the stronger multi-horizon rank predictor."""
import numpy as np
import torch

import experiment_dense_name_error_memory_prior_v1 as prior
from prepare_dense_shared_rank_reference_v1 import OUT as REFERENCE

rel = prior.rel
OUT = REFERENCE.parent / "dense_live_error_on_shared_ranks_v1"
ARMS = ("live_memory_innovation",)
USES_FORECAST = True


def train(end):
    if not (prior.OUT/f"prior_{end}.npz").exists():
        prior.train(end)
    return np.array([end])


def views(raw,grid,code,bank,p):
    return prior.views(raw,grid,code,bank,p)[:,:,2:3]


if __name__=="__main__":
    torch.set_num_threads(1)
    rel.main(OUT,ARMS,train,views,with_prediction=True,rolling_reference=REFERENCE)
