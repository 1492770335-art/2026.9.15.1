# -*- coding: utf-8 -*-
"""TC-32 Gate R0: reproduce official Score rounding for EXP-0020 routing (Fold A/B).
Reads official evaluate.py semantics (score = round(unrounded_corr @ weights, 4); displayed
corr0/1/2 individually rounded). Recomputes from frozen routing commit via batched inference.
"""
import importlib.util
import json
import os
import sys

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
sys.path.insert(0, os.path.join("project_restart", "src", "models"))
from script.evaluate import Evaluator  # noqa: E402
from gru_strong import load_block, make_batch  # noqa: E402
from src.eval.metrics import pooled_pearson_raw  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0020")
K = 16


def load_models(commit_dir, device="cuda:0"):
    spec = importlib.util.spec_from_file_location("netw_r0", os.path.join(commit_dir, "network.py"),
                                                  submodule_search_locations=[os.path.abspath(commit_dir)])
    netw = importlib.util.module_from_spec(spec)
    sys.modules["netw_r0"] = netw
    spec.loader.exec_module(netw)
    models = []
    for label in range(3):
        cfg = json.load(open(os.path.join(commit_dir, "conf", f"Model_{label}.json"), encoding="utf-8"))
        m = getattr(netw, cfg["model"])(cfg).to(device)
        m.load_state_dict(torch.load(os.path.join(commit_dir, "model", f"Model_{label}.pth"),
                                     map_location=device, weights_only=True))
        m.eval()
        models.append(m)
    return models


def add_market(X):
    T = X.reshape(-1, 50, 100)
    m = T.mean(axis=1, keepdims=True)
    s = T.std(axis=1, keepdims=True, ddof=1)
    m = np.broadcast_to(m, T.shape).reshape(-1, 100)
    s = np.broadcast_to(s, T.shape).reshape(-1, 100)
    return np.concatenate([X, m, s], axis=1).astype(np.float32)


def batched_predict(models, dates, device="cuda:0"):
    X, y, tid = load_block(dates)
    Xm = add_market(X)
    preds = []
    with torch.no_grad():
        for m in models:
            o = []
            for s in range(0, len(X), 8192):
                I = np.arange(s, min(s + 8192, len(X)))
                if hasattr(m, "members"):
                    Xg = torch.from_numpy(make_batch(X, I, tid, K)).to(device)
                    out = torch.stack([mem(Xg) for mem in m.members], 0).mean(0)
                else:
                    Xg = torch.from_numpy(make_batch(Xm, I, tid, K)).to(device)
                    out = m(Xg)
                o.append(out.cpu().numpy())
            preds.append(np.concatenate(o))
    return np.stack(preds, axis=1).reshape(-1, 3), y


def official_rounding(corr):
    w = np.array([0.2, 0.3, 0.5])
    return {
        "corr0_disp": round(float(corr[0]), 4),
        "corr1_disp": round(float(corr[1]), 4),
        "corr2_disp": round(float(corr[2]), 4),
        "official_score": round(float(corr @ w), 4),          # unrounded corr then round
        "naive_from_displayed": round(float(np.round(corr, 4) @ w), 4),  # wrong order
        "raw_weighted": float(corr @ w),
    }


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    res = {}
    for fold in ("A", "B"):
        dates = [8, 9, 10, 11] if fold == "A" else [12, 13, 14, 15]
        commit = os.path.join(OUT, f"commit_routing_{fold}")
        models = load_models(commit, device)
        P, y = batched_predict(models, dates, device)
        corr = pooled_pearson_raw(P, y)
        res[fold] = {"unrounded_corr": corr.tolist(), **official_rounding(corr)}
        # official evaluator authoritative
        ev = Evaluator(DATA, commit, device)
        res[fold]["official_evaluator"] = ev.evaluate(date=dates)
        print(f"[{fold}] unrounded_corr={corr} official_eval={res[fold]['official_evaluator']}", flush=True)
        print(f"[{fold}] official_rounding={official_rounding(corr)}", flush=True)
    out = os.path.join(OUT, "r0_rounding_audit.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    print("written:", out)
