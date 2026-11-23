"""Parameter-matched pure causal TCN replacements for PeerGRU400128.

Production inputs retain the existing fixed sixteen-step, day-reset, left-zero
padding. No padding is skipped or masked after input projection. The encoder
contains no recurrent state, batch normalization, dropout, or future context.
"""
import torch
from torch import nn
from torch.nn import functional as F

import experiment_dynamic_peer_gru as peer

DILATIONS = {'dilated': (1, 2, 4, 8), 'local': (1, 1, 1, 1)}
PARAMETERS = 203650
WINDOW = 16


class CausalGatedBlock(nn.Module):
    def __init__(self, dilation):
        super().__init__()
        self.dilation = int(dilation)
        self.norm = nn.LayerNorm(128)
        self.expand = nn.Linear(128, 192)
        self.depthwise = nn.Conv1d(192, 192, kernel_size=2, dilation=self.dilation, groups=192)
        self.project = nn.Linear(96, 128)

    def forward(self, x):
        # LayerNorm acts only on the channel dimension of [B,T,128].
        expanded = self.expand(self.norm(x)).transpose(1, 2)
        temporal = self.depthwise(F.pad(expanded, (self.dilation, 0))).transpose(1, 2)
        content, gate = temporal.chunk(2, dim=-1)
        return x + self.project(F.silu(content) * torch.sigmoid(gate))


class PeerTCN(nn.Module):
    def __init__(self, mode='dilated'):
        super().__init__()
        if mode not in DILATIONS:
            raise ValueError(f'Unknown TCN mode: {mode}')
        self.mode = mode
        # Keep this construction order aligned with the paired encoder family.
        # Only the reference prediction heads are registered on this module.
        reference = peer.PeerGRU()
        self.head1 = reference.head1
        self.head2 = reference.head2
        self.input = nn.Linear(400, 128)
        with torch.no_grad():
            self.input.weight[:, 300:] = 0
        self.blocks = nn.ModuleList([CausalGatedBlock(dilation) for dilation in DILATIONS[mode]])

    def encode_sequence(self, x):
        """Encode [B,T,400] to [B,T,128], with only causal per-frame operations."""
        if x.ndim != 3 or x.shape[0] < 1 or x.shape[1] < 1 or x.shape[2] != 400:
            raise ValueError('Expected a nonempty tensor shaped [batch,time,400]')
        hidden = self.input(x)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def forward(self, x):
        last = self.encode_sequence(x)[:, -1]
        return torch.cat((self.head1(last), self.head2(last)), dim=-1)


def preflight():
    """Synthetic CPU checks; no dataset, GPU, or output files are used."""
    torch.set_num_threads(2)
    generator = torch.Generator(device='cpu').manual_seed(2027)
    models = {}
    for mode in DILATIONS:
        torch.manual_seed(42)
        models[mode] = PeerTCN(mode)
    assert sum(p.numel() for p in models['dilated'].parameters()) == PARAMETERS
    assert sum(p.numel() for p in models['local'].parameters()) == PARAMETERS
    assert set(models['local'].state_dict()) == set(models['dilated'].state_dict())
    for name, value in models['local'].state_dict().items():
        torch.testing.assert_close(value, models['dilated'].state_dict()[name], atol=0, rtol=0)
    torch.manual_seed(42)
    reference = peer.PeerGRU()
    for head_name in ('head1', 'head2'):
        for name, value in getattr(reference, head_name).state_dict().items():
            torch.testing.assert_close(value, getattr(models['local'], head_name).state_dict()[name], atol=0, rtol=0)
    assert not any(isinstance(m, nn.GRU) for m in models['dilated'].modules())
    report = {'parameters': PARAMETERS, 'same_initial_state_dict_across_modes': True,
              'reference_heads_initialization_preserved': True, 'reference_gru_not_registered': True}

    receptive_fields = {}
    for mode, original in models.items():
        original.eval()
        model = original.double()
        x = .1 * torch.randn(3, 24, 400, dtype=torch.float64, generator=generator)
        perturbed = x.clone()
        perturbed[:, 10:] += torch.randn(3, 14, 400, dtype=torch.float64, generator=generator)
        with torch.no_grad():
            before = model.encode_sequence(x)
            after = model.encode_sequence(perturbed)
            torch.testing.assert_close(before[:, :10], after[:, :10], atol=0, rtol=0)
            together = model(x[:, -16:])
            split = torch.cat([model(x[:1, -16:]), model(x[1:, -16:])])
            torch.testing.assert_close(together, split, atol=1e-12, rtol=1e-10)
            independent = x[:, -16:].clone()
            independent[1:] += 100
            torch.testing.assert_close(model(independent)[:1], together[:1], atol=1e-12, rtol=1e-10)
        # Probe a non-saturated double-precision path to the oldest accessible tick.
        probe = (.1 * torch.randn(1, 20, 400, dtype=torch.float64, generator=generator)).requires_grad_()
        scalar = model(probe)[0, 1]
        gradient = torch.autograd.grad(scalar, probe)[0]
        per_tick = gradient[0].norm(dim=-1)
        width = 1 + sum(DILATIONS[mode])
        earliest = probe.shape[1] - width
        assert torch.count_nonzero(per_tick[:earliest]) == 0
        if not float(per_tick[earliest]) > 1e-16:
            raise AssertionError(f'Oldest receptive-field path is inactive for {mode}')
        assert torch.isfinite(per_tick).all()
        receptive_fields[mode] = {'ticks': width, 'earliest_input_gradient_norm': float(per_tick[earliest]),
                                  'outside_receptive_field_gradient_exactly_zero': True}
    report.update(future_perturbation_preserves_prefix=True, batch_split_and_independence=True,
                  receptive_fields=receptive_fields)

    # Independently build fixed windows versus a reset, left-zero input ring.
    # Recompute the encoder on all16 raw-feature positions, including padded zeros.
    day_frames = torch.randn(2, 19, 3, 400, dtype=torch.float64, generator=generator)
    day_frames[1] += .5
    for mode, model in models.items():
        windows = []
        for day in range(2):
            for tick in range(19):
                start = max(0, tick - WINDOW + 1)
                observed = day_frames[day, start:tick + 1].permute(1, 0, 2)
                windows.append(F.pad(observed, (0, 0, WINDOW - observed.shape[1], 0)))
        with torch.no_grad():
            batched = model(torch.cat(windows)).reshape(2, 19, 3, 2)
            for day in range(2):
                ring = torch.zeros(3, WINDOW, 400, dtype=torch.float64)
                for tick in range(19):
                    ring = torch.cat((ring[:, 1:], day_frames[day, tick, :, None, :]), dim=1)
                    expected_window = windows[day * 19 + tick]
                    torch.testing.assert_close(ring, expected_window, atol=0, rtol=0)
                    torch.testing.assert_close(model(ring), batched[day, tick], atol=1e-12, rtol=1e-10)
    report['fixed16_zero_padding_and_day_reset_input_ring_match_batch'] = True

    peer_gradient_norms = {}
    for mode, model in models.items():
        model.train()
        model.zero_grad(set_to_none=True)
        x = torch.randn(4, WINDOW, 400, dtype=torch.float64, generator=generator)
        target = torch.randn(4, 2, dtype=torch.float64, generator=generator)
        assert torch.count_nonzero(model.input.weight[:, 300:]) == 0
        changed_peer = x.clone()
        changed_peer[:, :, 300:] += torch.randn(4, WINDOW, 100, dtype=torch.float64, generator=generator)
        with torch.no_grad():
            torch.testing.assert_close(model(x), model(changed_peer), atol=0, rtol=0)
        loss = (model(x) - target).square().mean()
        assert torch.isfinite(loss)
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                raise AssertionError(f'Missing/nonfinite parameter gradient: {mode}.{name}')
        norm = float(model.input.weight.grad[:, 300:].norm())
        assert norm > 0
        peer_gradient_norms[mode] = norm
    report.update(zero_initialized_peer_columns_have_nonzero_loss_gradient=True,
                  peer_input_gradient_norm=peer_gradient_norms, all_parameter_gradients_finite=True)
    return report


if __name__ == '__main__':
    import json
    print(json.dumps(preflight(), indent=2, sort_keys=True))
