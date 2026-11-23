"""Transfer a walk-forward selected range expert innovation across overlapping horizons."""
import json
import numpy as np
import torch
import experiment_session_extrema_mixture_v1 as expert
import experiment_joint_horizon_completion_v1 as joint
import experiment_side_network_execution_fusion_v1 as fusion
import replay_verified_forecast_parent_v1 as replay
from experiment_causal_cross_horizon_ridge_v1 import ART, labels, measure

PARENT = ART / 'partial_horizon_curve_v1'
OUT = ART / 'range_horizon_transfer_v1'


def forecast(base, side, weight, transfer):
    delta = fusion.add(base, side, weight) - base
    shared = delta.mean(1, keepdims=True)
    return (base + delta + shared @ transfer[0] + (delta - shared) @ transfer[1]).astype('float32')


def main():
    OUT.mkdir(exist_ok=True)
    weight = np.load(expert.OUT / 'coef.npy')
    transfer, = joint.coefficient([('', '', expert.OUT.name)], joint.residual_covariances())
    np.savez(OUT / 'weights.npz', weight=weight, transfer=transfer)
    rows = []
    for d in range(16, 20):
        p = np.load(PARENT / f'pred_{d}.npy')
        q = forecast(p, np.load(expert.OUT / f'side_{d}.npy'), weight, transfer)
        np.save(OUT / f'pred_{d}.npy', q)
        rows.append((p.reshape(-1, 3), q.reshape(-1, 3), labels(d)))
    result = measure(rows)
    (OUT / 'results.json').write_text(json.dumps({'20': result}, indent=2))
    print(result, flush=True)
    if result['gain'] > 0:
        st = torch.load(expert.OUT / 'net_16.pth', map_location='cpu', weights_only=True)
        net = expert.restore(st).eval()
        net.load_state_dict(st)
        replay.main(PARENT, OUT, lambda raw, p: forecast(p, expert.infer(net, raw), weight, transfer))


if __name__ == '__main__':
    main()
