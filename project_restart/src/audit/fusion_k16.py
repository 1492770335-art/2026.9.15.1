# -*- coding: utf-8 -*-
"""EXP-0018B: K16 3-seed fixed equal-weight fusion (zero training, frozen weights).
seeds: 42 (EXP-0016 commit_K16_{A,B}), 43/44 (EXP-0017 commit_s{43,44}_K16_{A,B}).
Reproduce 6 single-seed scores; fuse per-target row-level 1:1:1; recompute pooled/daily/date8;
y2 prediction & residual pairwise corr; build K16 3-member ensemble commit + official streaming
latency & VRAM. No weight learning, no true lagged labels.
"""
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
sys.path.insert(0, os.path.join("project_restart", "src", "models"))
from script.evaluate import Evaluator  # noqa: E402
from gru_strong import load_block, make_batch, CFG  # noqa: E402
from src.eval.metrics import official_score  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
K = 16
CFG["window"] = K
OUT = os.path.join("project_restart", "experiments", "EXP-0018")
os.makedirs(OUT, exist_ok=True)

COMMITS = {
    "A": {42: os.path.join("project_restart", "experiments", "EXP-0016", "commit_K16_A"),
          43: os.path.join("project_restart", "experiments", "EXP-0017", "commit_s43_K16_A"),
          44: os.path.join("project_restart", "experiments", "EXP-0017", "commit_s44_K16_A")},
    "B": {42: os.path.join("project_restart", "experiments", "EXP-0016", "commit_K16_B"),
          43: os.path.join("project_restart", "experiments", "EXP-0017", "commit_s43_K16_B"),
          44: os.path.join("project_restart", "experiments", "EXP-0017", "commit_s44_K16_B")},
}
RECORDED = {"A": {42: 0.2390, 43: 0.2432, 44: 0.2468}, "B": {42: 0.2612, 43: 0.2582, 44: 0.2638}}
VALS = {"A": [8, 9, 10, 11], "B": [12, 13, 14, 15]}


def load_commit_models(commit_dir, device="cuda:0"):
    spec = importlib.util.spec_from_file_location("netw5", os.path.join(commit_dir, "network.py"),
                                                  submodule_search_locations=[os.path.abspath(commit_dir)])
    netw = importlib.util.module_from_spec(spec)
    sys.modules["netw5"] = netw
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


def predict_batched(models, dates, device="cuda:0"):
    X, y, tid = load_block(dates)
    dcol = (pl.scan_parquet(DATA).filter(pl.col("date").is_in(sorted(dates)))
            .sort(["date", "time", "code"]).select("date").collect()["date"].to_numpy())
    preds = []
    with torch.no_grad():
        for m in models:
            o = []
            for s in range(0, len(X), 8192):
                I = np.arange(s, min(s + 8192, len(X)))
                Xg = torch.from_numpy(make_batch(X, I, tid, K)).to(device)
                o.append(m(Xg).cpu().numpy())
            preds.append(np.concatenate(o))
    return np.stack(preds, axis=1).reshape(-1, 3), y, dcol


def build_ensemble_commit(fold):
    commit = os.path.join(OUT, f"commit_ens3_K16_{fold}")
    if os.path.exists(commit):
        shutil.rmtree(commit)
    os.makedirs(os.path.join(commit, "conf"))
    os.makedirs(os.path.join(commit, "model"))
    shutil.copy(os.path.join("project", "project", "script", "base.py"), os.path.join(commit, "base.py"))
    net = (
        "import torch\n"
        "from torch import nn\n"
        "from torch.nn import functional as F\n"
        "from .base import BaseModel\n"
        "class _GRU(nn.Module):\n"
        "    def __init__(self, hidden):\n"
        "        super().__init__()\n"
        "        self.gru = nn.GRU(100, hidden, batch_first=True)\n"
        "        self.output = nn.Linear(hidden, 1)\n"
        "    def forward(self, x):\n"
        "        h, _ = self.gru(x)\n"
        "        return self.output(h[:, -1])\n"
        "class GRUEnsemble(BaseModel):\n"
        "    def __init__(self, config):\n"
        "        super().__init__(config)\n"
        "        hidden = config['hidden']\n"
        "        self.window = config.get('window', 16)\n"
        "        self.members = nn.ModuleList([_GRU(hidden) for _ in range(3)])\n"
        "        self.hist = [None, None, None]\n"
        "    def reset(self):\n"
        "        self.hist = [None, None, None]\n"
        "    def predict(self, x):\n"
        "        outs = []\n"
        "        for k, m in enumerate(self.members):\n"
        "            cur = x[:, None].detach()\n"
        "            self.hist[k] = cur if self.hist[k] is None else torch.cat([self.hist[k], cur], dim=1)\n"
        "            self.hist[k] = self.hist[k][:, -self.window:]\n"
        "            h = F.pad(self.hist[k], (0, 0, self.window - self.hist[k].shape[1], 0))\n"
        "            outs.append(m(h))\n"
        "        return torch.stack(outs, dim=0).mean(dim=0)\n"
    )
    with open(os.path.join(commit, "network.py"), "w", encoding="utf-8") as f:
        f.write(net)
    return commit


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    res = {}
    for fold in ("A", "B"):
        dates = VALS[fold]
        seed_preds = {}
        res[fold] = {}
        for seed in (42, 43, 44):
            models = load_commit_models(COMMITS[fold][seed], device)
            P, y, dcol = predict_batched(models, dates, device)
            o = official_score(P, y)
            rec = RECORDED[fold][seed]
            res[fold][f"seed{seed}"] = {"official": o, "recorded": rec,
                                        "reproduced": bool(abs(o["score"] - rec) < 0.0002),
                                        "daily": {int(dd): official_score(P[dcol == dd], y[dcol == dd]) for dd in dates}}
            seed_preds[seed] = P
            print(f"[{fold} s{seed}] batched={o['score']} recorded={rec} ok={abs(o['score']-rec)<0.0002}", flush=True)
        P_fus = (seed_preds[42] + seed_preds[43] + seed_preds[44]) / 3.0
        o_fus = official_score(P_fus, y)
        res[fold]["fusion_111"] = {
            "official": o_fus,
            "daily": {int(dd): official_score(P_fus[dcol == dd], y[dcol == dd]) for dd in dates},
        }
        if fold == "A":
            m = np.isin(dcol, [9, 10, 11])
            res[fold]["fusion_111"]["no_date8"] = official_score(P_fus[m], y[m])
        pc, rc = {}, {}
        for j, yn in enumerate(["y0", "y1", "y2"]):
            pa, pb, pcc = seed_preds[42][:, j], seed_preds[43][:, j], seed_preds[44][:, j]
            pc[yn] = {"42_43": float(np.corrcoef(pa, pb)[0, 1]), "42_44": float(np.corrcoef(pa, pcc)[0, 1]), "43_44": float(np.corrcoef(pb, pcc)[0, 1])}
            ea, eb, ec = y[:, j] - pa, y[:, j] - pb, y[:, j] - pcc
            rc[yn] = {"42_43": float(np.corrcoef(ea, eb)[0, 1]), "42_44": float(np.corrcoef(ea, ec)[0, 1]), "43_44": float(np.corrcoef(eb, ec)[0, 1])}
        res[fold]["prediction_pairwise_corr"] = pc
        res[fold]["residual_pairwise_corr"] = rc
        res[fold]["deltas"] = {
            "fusion_vs_s42": round(o_fus["score"] - RECORDED[fold][42], 4),
            "fusion_vs_best_single": round(o_fus["score"] - max(RECORDED[fold].values()), 4),
            "fusion_vs_seed_mean": round(o_fus["score"] - np.mean(list(RECORDED[fold].values())), 4),
        }
        print(f"[{fold}] fusion={o_fus} deltas={res[fold]['deltas']}", flush=True)
        # ensemble commit + official streaming (latency & VRAM)
        commit = build_ensemble_commit(fold)
        for label in range(3):
            ens_sd = {}
            for i, seed in enumerate((42, 43, 44)):
                p = os.path.join(COMMITS[fold][seed], "model", f"Model_{label}.pth")
                state = torch.load(p, map_location="cpu", weights_only=True)
                for k, v in state.items():
                    ens_sd[f"members.{i}.{k}"] = v
            torch.save(ens_sd, os.path.join(commit, "model", f"Model_{label}.pth"))
            with open(os.path.join(commit, "conf", f"Model_{label}.json"), "w", encoding="utf-8") as f:
                json.dump({"model": "GRUEnsemble", "hidden": CFG["hidden"], "window": K}, f)
        ev = Evaluator(DATA, commit, device)
        t0 = time.time()
        stream = ev.evaluate(date=dates[0])
        lat = time.time() - t0
        res[fold]["streaming_check"] = {"date": dates[0], "result": stream,
                                        "batched_same_day": res[fold]["fusion_111"]["daily"][dates[0]]["score"],
                                        "latency_sec": round(lat, 1),
                                        "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1)}
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        print(f"[{fold}] streaming date{dates[0]}={stream} latency={lat:.1f}s", flush=True)
    with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    print("done", flush=True)
