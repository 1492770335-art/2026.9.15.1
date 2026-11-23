"""Replay the stock-relative intraday calibration from the same raw panel as the base."""
import json
import numpy as np
import replay_curve_quantile_market_v1 as replay
import experiment_intraday_relative_calibration_v1 as calibrated

def predict(raw, opening):
    p=replay.build(raw,opening).astype(np.float32).astype(float)
    coef=np.load(calibrated.OUT/'coef.npy')
    return p+np.einsum('tskj,kj->tsk',calibrated.adaptive(raw,p),coef)

def main():
    reference=json.loads((calibrated.OUT/'results.json').read_text())['20']['score']
    replay.main(predict,calibrated.OUT,reference)

if __name__=='__main__':main()
