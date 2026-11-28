"""Locations shared by the released layers and the trained base release."""
import os
from pathlib import Path
import numpy as np

ROOT = Path(os.environ['TULIN_RUN_DIR']).resolve()
PRIOR = Path(os.environ['TULIN_PRIOR_SRC']).resolve()
CACHE = Path(os.environ.get('TULIN_CACHE', PRIOR/'artifacts/joint_transformer/cache')).resolve()
TAPE = Path(os.environ.get('TULIN_ROLLING', ROOT/'rolling')).resolve()
GEO = PRIOR/'artifacts/project_restart_generalization/compounded_return_audit_v1/model.npz'
UNITS = np.array([.160943/.558942, .160943/.272164, 1.])


def released(day, raw):
    """Retrieve the pre-five-tick forecast saved by the feature-only base forward pass."""
    with np.load(ROOT/'artifacts/dense_five_tick_verified_reference_v1'/f'previous_{day}.npz') as data:
        for key in ('full', 'prefix'):
            if len(data[key]) == len(raw):
                return data[key].copy()
    raise ValueError('No previous release prediction for the requested day and length')
