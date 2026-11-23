"""Complex diagonal LRU replacement for the fixed-window PeerGRU.

Recurrence and initialization follow Appendix A of Orvieto et al., ICML 2023:
https://proceedings.mlr.press/v202/orvieto23a/orvieto23a.pdf
All trainable parameters and scan operations are real-valued. There is no
persistent recurrent state: every supplied window starts from zero state.
"""
import json
import math

import torch
from torch import nn
from torch.nn import functional as F

import experiment_dynamic_peer_gru as peer


class ComplexDiagonalLRU(nn.Module):
    def __init__(self, d_model=128, state_size=72):
        super().__init__()
        self.d_model = d_model
        self.state_size = state_size
        radius = torch.sqrt(torch.rand(state_size) * (.99 ** 2 - .8 ** 2) + .8 ** 2)
        phase = (torch.rand(state_size) * (2 * math.pi)).clamp_min(1e-6)
        self.nu_log = nn.Parameter(torch.log(-torch.log(radius)))
        self.theta_log = nn.Parameter(torch.log(phase))
        self.gamma_log = nn.Parameter(.5 * torch.log1p(-radius.square()))
        self.B_r = nn.Parameter(torch.randn(state_size, d_model) / math.sqrt(2 * d_model))
        self.B_i = nn.Parameter(torch.randn(state_size, d_model) / math.sqrt(2 * d_model))
        self.C_r = nn.Parameter(torch.randn(d_model, state_size) / math.sqrt(state_size))
        self.C_i = nn.Parameter(torch.randn(d_model, state_size) / math.sqrt(state_size))
        self.Dskip = nn.Parameter(torch.randn(d_model))

    def eigenvalues(self):
        radius = torch.exp(-torch.exp(self.nu_log))
        phase = torch.exp(self.theta_log)
        return radius * torch.cos(phase), radius * torch.sin(phase)

    def _drive(self, u):
        if u.ndim != 3 or u.shape[-1] != self.d_model or u.shape[1] == 0:
            raise ValueError('Expected a nonempty B,T,d_model input')
        gamma = self.gamma_log.exp()
        return F.linear(u, self.B_r) * gamma, F.linear(u, self.B_i) * gamma

    def scan_states(self, u):
        """Inclusive doubling scan, four stages for a 16-step window."""
        h_r, h_i = self._drive(u)
        power_r, power_i = self.eigenvalues()
        offset = 1
        while offset < u.shape[1]:
            # Previous-stage arrays are read before producing either new array.
            previous_r = F.pad(h_r[:, :-offset], (0, 0, offset, 0))
            previous_i = F.pad(h_i[:, :-offset], (0, 0, offset, 0))
            h_r, h_i = (h_r + power_r * previous_r - power_i * previous_i,
                        h_i + power_r * previous_i + power_i * previous_r)
            power_r, power_i = (power_r.square() - power_i.square(),
                                2 * power_r * power_i)
            offset *= 2
        return h_r, h_i

    def _output(self, u, h_r, h_i):
        return F.linear(h_r, self.C_r) - F.linear(h_i, self.C_i) + self.Dskip * u

    def forward(self, u):
        return self._output(u, *self.scan_states(u))

    def sequential_reference(self, u):
        """Slow zero-initial-state reference for CPU tests, never training."""
        drive_r, drive_i = self._drive(u)
        lambda_r, lambda_i = self.eigenvalues()
        previous_r = torch.zeros_like(drive_r[:, 0])
        previous_i = torch.zeros_like(drive_i[:, 0])
        states_r, states_i = [], []
        for t in range(u.shape[1]):
            previous_r, previous_i = (
                lambda_r * previous_r - lambda_i * previous_i + drive_r[:, t],
                lambda_r * previous_i + lambda_i * previous_r + drive_i[:, t])
            states_r.append(previous_r)
            states_i.append(previous_i)
        return self._output(u, torch.stack(states_r, dim=1), torch.stack(states_i, dim=1))


class LRUBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(128)
        self.lru = ComplexDiagonalLRU(128, 72)
        self.gate = nn.Linear(128, 200)
        self.output = nn.Linear(100, 128)

    def _finish(self, x, recurrent):
        a, b = self.gate(recurrent).chunk(2, dim=-1)
        return x + self.output(F.silu(a) * torch.sigmoid(b))

    def forward(self, x):
        return self._finish(x, self.lru(self.norm(x)))

    def sequential_reference(self, x):
        return self._finish(x, self.lru.sequential_reference(self.norm(x)))


class PeerLRU(nn.Module):
    def __init__(self):
        super().__init__()
        # Consume the same initial RNG sequence and retain the original heads.
        reference = peer.PeerGRU()
        self.head1 = reference.head1
        self.head2 = reference.head2
        self.input = nn.Linear(400, 128)
        with torch.no_grad():
            self.input.weight[:, 300:] = 0
        self.blocks = nn.ModuleList([LRUBlock(), LRUBlock()])

    def encode_sequence(self, x):
        if x.ndim != 3 or x.shape[-1] != 400 or x.shape[1] == 0:
            raise ValueError('Expected a nonempty B,T,400 history window')
        hidden = self.input(x)
        for block in self.blocks:
            hidden = block(hidden)
        return hidden

    def forward(self, x):
        last = self.encode_sequence(x)[:, -1]
        return torch.cat([self.head1(last), self.head2(last)], dim=-1)

    def encode_sequence_reference(self, x):
        """Sequential test implementation, unused by forward."""
        hidden = self.input(x)
        for block in self.blocks:
            hidden = block.sequential_reference(hidden)
        return hidden


def preflight():
    """Synthetic CPU tests with no data access, file output, or RNG changes."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(42)
        reference = peer.PeerGRU()
        torch.manual_seed(42)
        model = PeerLRU()
        for name in ('head1', 'head2'):
            for key, value in getattr(reference, name).state_dict().items():
                torch.testing.assert_close(value, getattr(model, name).state_dict()[key], atol=0, rtol=0)
        parameters = sum(p.numel() for p in model.parameters())
        if parameters != 203970 or sum(p.numel() for p in reference.parameters()) != 203778:
            raise AssertionError('Unexpected parameter count')
        if any(isinstance(module, nn.GRU) for module in model.modules()):
            raise AssertionError('GRU was not replaced')
        torch.testing.assert_close(model.input.weight[:, 300:], torch.zeros(128, 100), atol=0, rtol=0)
        if any(p.is_complex() for p in model.parameters()):
            raise AssertionError('Expected real-valued trainable tensors')

        layer = ComplexDiagonalLRU().double()
        u = torch.randn(2, 16, 128, dtype=torch.float64, requires_grad=True)
        probe = torch.randn_like(u)
        parallel = layer(u)
        serial = layer.sequential_reference(u)
        torch.testing.assert_close(parallel, serial, atol=2e-12, rtol=2e-11)
        scan_gradient = torch.autograd.grad((parallel * probe).sum(), (u,) + tuple(layer.parameters()))
        serial_gradient = torch.autograd.grad((serial * probe).sum(), (u,) + tuple(layer.parameters()))
        max_gradient_error = 0.
        for left, right in zip(scan_gradient, serial_gradient):
            torch.testing.assert_close(left, right, atol=2e-10, rtol=2e-10)
            max_gradient_error = max(max_gradient_error, float((left - right).abs().max()))
        torch.testing.assert_close(layer(torch.zeros_like(u)), torch.zeros_like(u), atol=0, rtol=0)
        for length in (1, 7):
            torch.testing.assert_close(layer(u[:, :length]), layer.sequential_reference(u[:, :length]), atol=2e-12, rtol=2e-11)

        model = model.double()
        x = torch.randn(3, 16, 400, dtype=torch.float64)
        with torch.no_grad():
            encoded = model.encode_sequence(x)
            sequential = model.encode_sequence_reference(x)
            torch.testing.assert_close(encoded, sequential, atol=2e-12, rtol=2e-11)
            changed = x.clone()
            changed[:, 7:] = torch.randn_like(changed[:, 7:]) * 10
            torch.testing.assert_close(encoded[:, :7], model.encode_sequence(changed)[:, :7], atol=2e-12, rtol=2e-11)
            torch.testing.assert_close(encoded[:, :7], model.encode_sequence(x[:, :7]), atol=2e-12, rtol=2e-11)
            together = model(x)
            split = torch.cat([model(x[i:i + 1]) for i in range(len(x))])
            torch.testing.assert_close(together, split, atol=2e-12, rtol=2e-11)
            model(torch.randn_like(x))
            replay = model(x)
            torch.testing.assert_close(together, replay, atol=0, rtol=0)
            changed = x.clone()
            changed[:, :, 300:] = torch.randn_like(changed[:, :, 300:]) * 10
            torch.testing.assert_close(together, model(changed), atol=0, rtol=0)
        radii = []
        for block in model.blocks:
            real, imaginary = block.lru.eigenvalues()
            radius = torch.sqrt(real.square() + imaginary.square())
            if not torch.all((radius > 0) & (radius < 1)):
                raise AssertionError('Unstable recurrent eigenvalues')
            radii.append([float(radius.min().detach()), float(radius.max().detach())])
        if list(model.buffers()):
            raise AssertionError('Unexpected persistent recurrent buffers')

        # Three ordinary clipped SGD steps on synthetic data, without importing
        # an optimizer backend that may create compilation/temp cache files.
        model = model.float()
        x = x.float()
        target = torch.randn(3, 2)
        losses = []
        graph_gradient = None
        for step in range(3):
            model.zero_grad(set_to_none=True)
            loss = (model(x) - target).square().mean()
            loss.backward()
            if not torch.isfinite(loss) or any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise AssertionError('Nonfinite training loss or gradient')
            if step == 0:
                graph_gradient = float(model.input.weight.grad[:, 300:].norm())
                if graph_gradient <= 0:
                    raise AssertionError('Zero-initialized graph projection cannot learn')
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            with torch.no_grad():
                for parameter in model.parameters():
                    parameter.add_(parameter.grad, alpha=-1e-3)
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise AssertionError('Nonfinite optimizer update')
            losses.append(float(loss.detach()))
        return {'parameters': parameters, 'baseline_parameters': 203778, 'matched_initial_heads': True,
                'real_parameters_complex_diagonal_recurrence': True, 'scan_stages_at_window16': 4,
                'scan_reference_max_abs_error': float((parallel - serial).abs().max().detach()),
                'scan_reference_input_and_parameter_gradient_max_abs_error': max_gradient_error,
                'full_model_matches_sequential_reference': True, 'causality_and_prefix': True,
                'zero_initial_state': True, 'window_replay_without_persistent_state': True,
                'batch_independent': True, 'initial_eigenvalue_radius_ranges': radii,
                'graph_projection_initially_zero': True, 'graph_projection_first_gradient_norm': graph_gradient,
                'synthetic_sgd_losses': losses}


if __name__ == '__main__':
    print(json.dumps(preflight()))
