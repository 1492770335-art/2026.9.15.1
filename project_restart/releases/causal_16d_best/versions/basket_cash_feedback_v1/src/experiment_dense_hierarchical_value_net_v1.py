"""Predict each stock's deviation from similar names and their group return with a shared nonlinear encoder."""

from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import experiment_dense_future_expert_value_v1 as future
import experiment_dense_peer_shrunk_field_prices_v1 as groups

ref, rel, panel = future.ref, future.rel, future.panel
OUT = ref.base.ROOT / "artifacts/dense_hierarchical_value_net_v1"
ARMS = ("single_name", "group_and_name", "future_expert_group")
USES_FORECAST = True


class ValueNet(nn.Module):
    def __init__(self, width, peers):
        super().__init__()
        self.register_buffer("peers", torch.tensor(peers, dtype=torch.float32))
        self.encoder = nn.Sequential(
            nn.Linear(width, 32),
            nn.LayerNorm(32),
            nn.SiLU(),
            nn.Dropout(0.12),
            nn.Linear(32, 24),
            nn.SiLU(),
        )
        self.query = nn.Linear(24, 8, bias=False)
        self.key = nn.Linear(24, 8, bias=False)
        self.local = nn.Linear(24, 1)
        self.group = nn.Sequential(nn.Linear(48, 16), nn.SiLU(), nn.Linear(16, 2))
        self.deviation = nn.Sequential(
            nn.Linear(48, 24), nn.SiLU(), nn.Dropout(0.10), nn.Linear(24, 2)
        )

    def forward(self, x):
        h = self.encoder(x)
        affinity = self.query(h) @ self.key(h).transpose(1, 2) / np.sqrt(8.0)
        graph = (affinity.clamp(-3, 3) + self.peers.clamp_min(1e-12).log()).softmax(-1)
        neighbors = graph @ h
        common = h.mean(1, keepdim=True).expand_as(h)
        group = self.group(torch.cat((neighbors, common), -1))
        own = self.deviation(torch.cat((h, h - neighbors), -1))
        return self.local(h), group, own


def features(raw, grid, code, p):
    x = panel.book(raw, grid, code)
    p = p - p.mean(1, keepdims=True)
    return np.concatenate((x, p.astype(np.float32)), 2)


def collect():
    OUT.mkdir(exist_ok=True)
    code = rel.fields.parent.codec()
    for d in range(4, 16):

        def load(k):
            return np.load(ref.base.CACHE / f"day_{d:02d}_{k}.npy", mmap_mode="r")

        raw, grid, y = load("x"), load("time").astype(int), load("y").astype(float)
        p = future.forecast(d).astype(float)
        p -= p.mean(1, keepdims=True)
        y -= y.mean(1, keepdims=True)
        scale = ((p * y).sum((0, 1)) / (p * p).sum((0, 1))).clip(0.2, 4)
        t = np.concatenate(
            [
                np.arange(a + 40 + d % 5, b - 61, 6)
                for a, b in rel.fields.parent.sessions(grid)
            ]
        )
        medium = sum(p[t + h, :, 1] for h in (0, 20, 40)) * ref.base.UNITS[1] * scale[1]
        true = y[t, :, 2] / scale[2] - p[t, :, 2]
        continuation = medium / scale[2] - p[t, :, 2]
        np.savez(
            OUT / f"day_{d:02d}.npz",
            x=features(raw, grid, code, p)[t].astype(np.float16),
            y=np.stack((true, continuation), -1).astype(np.float32),
        )


def train(days):
    torch.manual_seed(71843)
    peers = np.load(groups.OUT / "peers.npy")
    records = []
    for d in range(4, days):
        with np.load(OUT / f"day_{d:02d}.npz") as f:
            records.append((f["x"], f["y"]))
    x = torch.as_tensor(
        np.concatenate([v[0] for v in records]), device="cuda", dtype=torch.float32
    )
    y = torch.as_tensor(
        np.concatenate([v[1] for v in records]), device="cuda", dtype=torch.float32
    )
    unit = x.square().mean((0, 1)).sqrt().clamp_min(0.03)
    x = (x / unit).clamp(-5, 5)
    scale = y.square().mean((0, 1)).sqrt().clamp_min(0.03)
    y = (y / scale).clamp(-4, 4)
    net = ValueNet(x.shape[-1], peers).cuda()
    opt = torch.optim.AdamW(net.parameters(), lr=0.0015, weight_decay=0.035)
    ygroup = net.peers @ y
    yown = y - ygroup
    for _ in range(5):
        for idx in torch.randperm(len(x), device="cuda").split(48):
            one, group, own = net(x[idx])
            loss = (
                F.smooth_l1_loss(one[..., 0], y[idx, :, 0])
                + F.smooth_l1_loss(group, ygroup[idx])
                + F.smooth_l1_loss(own, yown[idx])
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
    torch.save(dict(state=net.state_dict(), unit=unit.cpu()), OUT / f"net_{days}.pt")
    print(f"Fitted on {days - 4} rolling forecast dates through day {days}", flush=True)
    return np.array([days])


@torch.no_grad()
def views(raw, grid, code, bank, p):
    state = torch.load(
        OUT / f"net_{int(bank[0])}.pt", map_location="cpu", weights_only=True
    )
    net = ValueNet(len(state["unit"]), np.load(groups.OUT / "peers.npy"))
    net.load_state_dict(state["state"])
    net = net.cuda().eval()
    x = features(raw, grid, code, p)
    out = []
    for start in range(0, len(x), 64):
        v = torch.from_numpy(x[start : start + 64]).cuda()
        one, group, own = net((v / state["unit"].cuda()).clamp(-5, 5))
        y = torch.cat((one, group + own), -1)
        out.append((y - y.mean(1, keepdim=True)).cpu().numpy())
    return np.concatenate(out)


if __name__ == "__main__":
    collect()
    rel.main(OUT, ARMS, train, views, with_prediction=True)
