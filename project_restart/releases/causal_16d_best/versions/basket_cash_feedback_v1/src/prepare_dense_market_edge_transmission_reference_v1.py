"""Training-fold predictions after stage 31."""
import numpy as np
import experiment_dense_market_edge_transmission_v1 as model
from version_paths import ROOT

OUT = ROOT / "artifacts/dense_market_edge_transmission_verified_reference_v1"


def read_p(day):
    return model.read_p(day) if day<6 else np.load(OUT/f'rolling_{day:02d}.npy')
