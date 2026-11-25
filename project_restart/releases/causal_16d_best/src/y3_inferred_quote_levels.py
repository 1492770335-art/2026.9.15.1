"""Decode two symmetric quote levels and a signed price offset from the anonymous price grid."""
from functools import lru_cache

import numpy as np

import y3_round_price_coordinates as grid

trial = grid.trial


def decode(field, level, offset, scale, positive, k=None):
    if k is None:
        k = grid.calibration()
    unit = 1/(k*level.clip(10))
    implied = (field-offset)/(unit*scale)
    count = np.rint(implied)
    aligned = level%2+2*np.rint((implied-level%2)/2)
    if positive:
        aligned = np.maximum(aligned, 2-level%2)
    return count.astype(int), aligned.astype(int), implied


@lru_cache(maxsize=1)
def calibration():
    a, b = [], []
    for d in range(4):
        raw = trial.clock.book.rawday(d)
        level, valid = grid.coordinates(raw)
        t = np.arange(240+d%13, len(raw), 13)
        a.append(raw[t][:,:,[10,40]][valid[t]].astype(float))
        b.append(level[t][valid[t]])
    fields, level = np.concatenate(a), np.concatenate(b)
    result = []
    for j, start in enumerate(((-.02028,.574),(-1.240,.965))):
        offset, scale = start
        for _ in range(6):
            _, aligned, implied = decode(fields[:,j], level, offset, scale, j==1)
            ok = abs(aligned-implied)<.32
            feature = aligned[ok]/(grid.calibration()*level[ok])
            mat = np.column_stack((np.ones(len(feature)),feature))
            offset,scale = np.linalg.lstsq(mat, fields[ok,j], rcond=1e-10)[0]
        result.append((offset,scale))
    return result


def infer(raw, k=None, coefficients=None):
    if k is None:
        k = grid.calibration()
    if coefficients is None:
        coefficients = calibration()
    level, reliable = grid.coordinates(raw,k)
    found = []
    for j,(i,(offset,scale)) in enumerate(zip((10,40),coefficients)):
        _, n, implied = decode(raw[:,:,i].astype(float),level,offset,scale,j==1,k=k)
        found.append(n)
        reliable &= abs(implied-n)<.35
    displacement, halfwidth=found
    return level,halfwidth,displacement,reliable
