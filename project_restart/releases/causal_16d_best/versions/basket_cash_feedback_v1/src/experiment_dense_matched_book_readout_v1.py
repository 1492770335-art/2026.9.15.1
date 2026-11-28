"""Fit return contrasts between separated observations of the same stock and trading session."""
from functools import lru_cache

import numpy as np
import torch

import experiment_phase_future_book_v1 as phase
import experiment_dense_path_skill_calibration_v1 as path
from prepare_dense_midday_expiry_reference_v1 import OUT as REFERENCE

onset, rel = path.onset, path.rel
OUT = REFERENCE.parent / "dense_matched_book_readout_v1"
ARMS = ("absolute_hidden", "paired_hidden", "matched_history_contrast")
LAGS = (64, 192, 384)


@lru_cache(maxsize=1)
def encoder():
    state = torch.load(OUT / "net_4.pth", map_location="cpu", weights_only=True)
    net = phase.BookNet(state["current.0.weight"].shape[1]).cuda().eval()
    net.load_state_dict(state)
    return net


@torch.inference_mode()
def hidden(raw, ticks):
    net = encoder()
    values, vol = phase.spline.features(raw, ticks)
    blocks = []
    for start in range(0, len(ticks), 64):
        batch = torch.from_numpy(values[start:start+64]).cuda().float()
        h = net.current(batch) + net.stock
        norm = net.norm(h)
        h = h + net.attn(norm, norm, norm, need_weights=False)[0]
        z = net.factors(h)
        factors = net.port.T @ (net.port @ z)
        local = net.reg(torch.cat((h, factors, z.mean(1, keepdim=True).expand_as(z)), dim=2))
        blocks.append(local.cpu().numpy())
    v = (.45+.55*vol).clip(.55, 3.)
    x = np.concatenate(blocks) * v[..., None]
    return x - x.mean(1, keepdims=True), v


def similarity(p, v, t, past):
    delta = ((p[t] - p[past]) / .5).clip(-6, 6)
    risk = np.log(v[t] / v[past])
    return 1 / (1 + (delta**2).mean(-1) + 2*risk**2)


def collect():
    OUT.mkdir(exist_ok=True)
    if not (OUT / "net_4.pth").exists():
        phase.train(4, out=OUT)
    for day in range(4, 16):
        raw, grid, y = [onset.settled.load(day, k) for k in ("x", "time", "y")]
        p = (np.load(REFERENCE/f"rolling_{day:02d}.npy") if day >= 6
             else onset.settled.head.ranks.band.pairs.net.future.forecast(day))
        current = p - p.mean(1, keepdims=True)
        label = y - y.mean(1, keepdims=True)
        slope = ((current*label).sum((0,1)) / (current**2).sum((0,1))).clip(.3,3)
        err = label/slope - current
        gram = np.zeros((3,96,99))
        counts = np.zeros(3)
        for a,b in rel.fields.parent.sessions(grid):
            ticks = np.arange(a+day%8,b,8)
            x, v = hidden(raw, ticks)
            residual = err[ticks]
            def add(arm, features, targets, weights):
                left = features.reshape(-1,96)
                right = np.concatenate((features, targets),axis=-1).reshape(-1,99)
                gram[arm] += (left*weights.reshape(-1,1)).T @ right
                counts[arm] += weights.sum()
            add(0, x, residual, np.ones(x.shape[:2], np.float32))
            for lag in LAGS:
                t = np.arange(lag//8,len(ticks))
                past = t-lag//8
                dx,dy = x[t]-x[past],residual[t]-residual[past]
                add(1,dx,dy,np.ones(dx.shape[:2],np.float32))
                add(2,dx,dy,similarity(current[ticks],v,t,past))
        np.save(OUT/f"gram_{day:02d}.npy",gram/counts[:,None,None])


def train(end):
    gram = np.mean([np.load(OUT/f"gram_{d:02d}.npy") for d in range(4,end)],axis=0)
    bank = []
    for row in gram:
        cov = row[:,:96]
        bank.append(np.linalg.solve(cov + .35*np.diag(cov.diagonal().clip(.001)),row[:,96:]))
    return np.stack(bank)


def design(raw,grid,code,p,bank):
    out = np.zeros((*p.shape[:2],len(ARMS),3))
    current = p - p.mean(1,keepdims=True)
    for a,b in rel.fields.parent.sessions(grid):
        x,v = hidden(raw,np.arange(a,b))
        for j in range(3):
            out[a:b,:,j] = x @ bank[j]
        projected = out[a:b,:,2].copy()
        total = np.zeros_like(projected)
        count = np.zeros(projected.shape[:2])
        for lag in LAGS:
            t = np.arange(lag,b-a)
            w = similarity(current[a:b],v,t,t-lag)
            total[t] += w[...,None]*(projected[t]-projected[t-lag])
            count[t] += w
        contrast = total / count.clip(1e-8)[...,None]
        out[a:b,:,2] = contrast-contrast.mean(1,keepdims=True)
    return out


def predict(raw,grid,p,params):
    return path.corrected(p,design(raw,grid,params["code"],p,params["bank"]),params)


if __name__ == "__main__":
    collect()
    onset.rolling(OUT,ARMS,train,design,REFERENCE)
