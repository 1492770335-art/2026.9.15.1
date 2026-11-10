# -*- coding: utf-8 -*-
"""TC-44 Gate G0: three-target incremental-opportunity & complementarity audit (zero-training).
Uses frozen predictions only: EXP-0018 (K16 3-seed all-target), EXP-0021 (Market routing),
and y2 candidates (clip GRU100, onehot GRU134, K16+K4 multiscale). Computes raw corr,
Market-vs-K16 complementarity (pred corr + residual corr), y2-candidate redundancy. No training.
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
sys.path.insert(0, os.path.join("project_restart", "src", "models"))
from gru_strong import load_block, make_batch, CFG  # noqa: E402
from src.eval.metrics import pooled_pearson_raw  # noqa: E402

CFG["window"] = 16
K = 16
OUT = os.path.join("project_restart", "experiments", "EXP-0032")
os.makedirs(OUT, exist_ok=True)


def add_market(X):
    T = X.reshape(-1, 50, 100)
    m = T.mean(axis=1, keepdims=True)
    s = T.std(axis=1, keepdims=True, ddof=1)
    m = np.broadcast_to(m, T.shape).reshape(-1, 100)
    s = np.broadcast_to(s, T.shape).reshape(-1, 100)
    return np.concatenate([X, m, s], axis=1).astype(np.float32)


def build_onehot17(X):
    cont = X[:, :83].astype(np.float32)
    cat = X[:, 83:100].astype(np.int64)
    n = len(X)
    oh = np.zeros((n, 51), dtype=np.float32)
    for j in range(17):
        v = cat[:, j]
        rows = np.where((v >= 0) & (v <= 2))[0]
        oh[rows, j * 3 + v[rows]] = 1.0
    return np.concatenate([cont, oh], axis=1).astype(np.float32)


def load_commit_models(commit_dir, device):
    modname = "netw_" + os.path.basename(commit_dir)
    spec = importlib.util.spec_from_file_location(modname, os.path.join(commit_dir, "network.py"),
                                                  submodule_search_locations=[os.path.abspath(commit_dir)])
    netw = importlib.util.module_from_spec(spec)
    sys.modules[modname] = netw
    spec.loader.exec_module(netw)
    models = []
    for label in range(3):
        cfg = json.load(open(os.path.join(commit_dir, "conf", f"Model_{label}.json"), encoding="utf-8"))
        m = getattr(netw, cfg["model"])(cfg).to(device)
        m.load_state_dict(torch.load(os.path.join(commit_dir, "model", f"Model_{label}.pth"), map_location=device, weights_only=True))
        m.eval()
        models.append(m)
    return models


def predict_commit(models, X_raw, tid, device):
    """Return (N,3) predictions. Market members get 300-dim; K16 members 100-dim."""
    Xm = add_market(X_raw)
    preds = []
    with torch.no_grad():
        for m in models:
            is_market = m.members[0].gru.input_size == 300
            Xuse = Xm if is_market else X_raw
            outs = []
            for s in range(0, len(Xuse), 8192):
                I = np.arange(s, min(s + 8192, len(Xuse)))
                Xg = torch.from_numpy(make_batch(Xuse, I, tid, K)).to(device)
                outs.append(torch.stack([mem(Xg) for mem in m.members], 0).mean(0).cpu().numpy())
            preds.append(np.concatenate(outs))
    return np.stack(preds, axis=1).reshape(-1, 3)


def corr(a, b):
    a = a.astype(np.float64).ravel(); b = b.astype(np.float64).ravel()
    n = len(a)
    return float((a @ b / n - a.mean() * b.mean()) / (a.std() * b.std() + 1e-12))


def load_single(ckpt, model_cls, device):
    m = model_cls().to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    m.eval()
    return m


def predict_single(m, Xuse, tid, device):
    with torch.no_grad():
        outs = []
        for s in range(0, len(Xuse), 8192):
            I = np.arange(s, min(s + 8192, len(Xuse)))
            Xg = torch.from_numpy(make_batch(Xuse, I, tid, K)).to(device)
            outs.append(m(Xg).cpu().numpy())
    return np.concatenate(outs)


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    res = {}
    for fold in ("A", "B"):
        dates = [8, 9, 10, 11] if fold == "A" else [12, 13, 14, 15]
        X, y, tid = load_block(dates)
        y2 = y[:, 2:3]
        # K16 3-seed all-target (EXP-0018)
        k16 = load_commit_models(os.path.join("project_restart", "experiments", "EXP-0018", f"commit_ens3_K16_{fold}"), device)
        Pk = predict_commit(k16, X, tid, device)
        ck = pooled_pearson_raw(Pk, y)
        # Market routing (EXP-0021)
        mk = load_commit_models(os.path.join("project_restart", "experiments", "EXP-0021", f"commit_3seed_routing_{fold}"), device)
        Pm = predict_commit(mk, X, tid, device)
        cm = pooled_pearson_raw(Pm, y)
        # complementarity y0/y1: Market vs K16
        comp = {}
        for t, name in ((0, "y0"), (1, "y1")):
            pred_corr = corr(Pm[:, t], Pk[:, t])
            resid_k16 = y[:, t] - Pk[:, t]
            resid_mkt = y[:, t] - Pm[:, t]
            comp[name] = {"pred_corr_market_vs_k16": pred_corr,
                          "corr_market_pred_with_k16_residual": corr(Pm[:, t], resid_k16),
                          "corr_k16_pred_with_market_residual": corr(Pk[:, t], resid_mkt)}
        # y2 candidates vs frozen K16 3-seed y2
        from exp_tc39_c1 import GRU100  # clip model class (same as K16)
        from exp_tc41_k1 import GRU134
        from ms_k4k16 import MultiScaleK4GRU
        k16_y2 = Pk[:, 2]
        cand = {}
        for name, ckpt, cls, feat in [
            ("clip", f"project_restart/experiments/EXP-0027/c1_s42_{fold}_y2.pth", GRU100, lambda x: x),
            ("onehot", f"project_restart/experiments/EXP-0029/k1_s42_{fold}_y2.pth", GRU134, build_onehot17),
            ("k4k16", f"project_restart/experiments/EXP-0031/m1_s42_{fold}_y2.pth", MultiScaleK4GRU, lambda x: x),
        ]:
            m = load_single(ckpt, cls, device)
            p = predict_single(m, feat(X), tid, device)
            cand[name] = {"pred_corr_with_k16_3seed_y2": corr(p, k16_y2),
                          "corr_with_k16_3seed_residual": corr(p, y2.ravel() - k16_y2)}
        res[fold] = {"k16_3seed_raw_corr": ck.tolist(), "market_routing_raw_corr": cm.tolist(),
                     "k16_3seed_score": round(float(ck @ np.array([0.2, 0.3, 0.5])), 4),
                     "market_routing_score": round(float(cm @ np.array([0.2, 0.3, 0.5])), 4),
                     "complementarity": comp, "y2_candidates": cand}
        print(f"[{fold}] K16 raw corr={ck} score={res[fold]['k16_3seed_score']}; Market raw corr={cm} score={res[fold]['market_routing_score']}", flush=True)
        print(f"[{fold}] complementarity={json.dumps(comp)}", flush=True)
        print(f"[{fold}] y2_candidates={json.dumps(cand)}", flush=True)
    with open(os.path.join(OUT, "complementarity.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    print("done")
