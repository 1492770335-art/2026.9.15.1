"""Training-fold predictions after stage 29."""
import numpy as np
import experiment_dense_cyclic_excursion_null_v1 as model
from version_paths import ROOT

OUT = ROOT / "artifacts/dense_cyclic_excursion_verified_reference_v1"


def read_p(d):
    return model.read_p(d) if d<6 else np.load(OUT/f'rolling_{d:02d}.npy')
