from functools import lru_cache

from pathlib import Path

import numpy as np

from version_paths import ROOT

OUT=ROOT/'artifacts/dense_basket_release_verified_reference_v1'

ARCHIVE=ROOT/'artifacts/dense_basket_cash_feedback_v1'

ROLLING=ROOT/'artifacts/dense_basket_feedback_verified_reference_v1'

def read_p(day):
    if day<6:
        from prepare_dense_basket_feedback_reference_v1 import read_p as early
        return early(day)
    return np.load(OUT/f'rolling_{day:02d}.npy')

@lru_cache(maxsize=1)
def short_market_changes():
    from experiment_dense_relative_path_inputs_v1 import rolling_short_changes
    return rolling_short_changes()

def read_with_short_market(day):
    p=read_p(day)
    if 8<=day<16:
        p=p.copy()
        p[...,0]+=short_market_changes()[day][:,None]
    return p
