# -*- coding: utf-8 -*-
"""TC-31 / Gate H0: zero-training per-target model routing.
y0 = EXP-0019 MarketGRU seed42; y1 = EXP-0019 MarketGRU seed42;
y2 = EXP-0018 K16 3-seed equal ensemble (seeds 42/43/44).
Build combined commit (MarketGRU for label0/1, GRUEnsemble for label2), verify via OFFICIAL
Evaluator (pooled + daily + date8), full-date-range engineering regression (in-sample diagnostic),
latency & VRAM. Compare vs K16 3-seed all-target fusion (A 0.2495 / B 0.2691).
"""
import json
import os
import shutil
import sys
import time

import torch

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
from script.evaluate import Evaluator  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0020")
os.makedirs(OUT, exist_ok=True)

SRC = {
    "A": {
        "market": os.path.join("project_restart", "experiments", "EXP-0019", "commit_mkt_A"),
        "ens": os.path.join("project_restart", "experiments", "EXP-0018", "commit_ens3_K16_A"),
    },
    "B": {
        "market": os.path.join("project_restart", "experiments", "EXP-0019", "commit_mkt_B"),
        "ens": os.path.join("project_restart", "experiments", "EXP-0018", "commit_ens3_K16_B"),
    },
}
VALS = {"A": [8, 9, 10, 11], "B": [12, 13, 14, 15]}
FUSION_REF = {"A": 0.2495, "B": 0.2691}

NET = (
    "import torch\n"
    "from torch import nn\n"
    "from torch.nn import functional as F\n"
    "from .base import BaseModel\n"
    "class MarketGRU(BaseModel):\n"
    "    def __init__(self, config):\n"
    "        super().__init__(config)\n"
    "        hidden = config['hidden']\n"
    "        self.gru = nn.GRU(300, hidden, batch_first=True)\n"
    "        self.output = nn.Linear(hidden, 1)\n"
    "        self.window = config.get('window', 16)\n"
    "        self.history = None\n"
    "    def forward(self, x):\n"
    "        h, _ = self.gru(x)\n"
    "        return self.output(h[:, -1])\n"
    "    def reset(self):\n"
    "        self.history = None\n"
    "    def predict(self, x):\n"
    "        m = x.mean(0, keepdim=True)\n"
    "        s = x.std(0, keepdim=True)\n"
    "        xm = torch.cat([x, m.expand_as(x), s.expand_as(x)], dim=1)\n"
    "        cur = xm[:, None].detach()\n"
    "        self.history = cur if self.history is None else torch.cat([self.history, cur], dim=1)\n"
    "        self.history = self.history[:, -self.window:]\n"
    "        h = F.pad(self.history, (0, 0, self.window - self.history.shape[1], 0))\n"
    "        return self.forward(h)\n"
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


def build_routing_commit(fold):
    commit = os.path.join(OUT, f"commit_routing_{fold}")
    if os.path.exists(commit):
        shutil.rmtree(commit)
    os.makedirs(os.path.join(commit, "conf"))
    os.makedirs(os.path.join(commit, "model"))
    shutil.copy(os.path.join("project", "project", "script", "base.py"), os.path.join(commit, "base.py"))
    with open(os.path.join(commit, "network.py"), "w", encoding="utf-8") as f:
        f.write(NET)
    # y0, y1 from market; y2 from ensemble
    shutil.copy(os.path.join(SRC[fold]["market"], "model", "Model_0.pth"), os.path.join(commit, "model", "Model_0.pth"))
    shutil.copy(os.path.join(SRC[fold]["market"], "model", "Model_1.pth"), os.path.join(commit, "model", "Model_1.pth"))
    shutil.copy(os.path.join(SRC[fold]["ens"], "model", "Model_2.pth"), os.path.join(commit, "model", "Model_2.pth"))
    for label, cls in ((0, "MarketGRU"), (1, "MarketGRU"), (2, "GRUEnsemble")):
        with open(os.path.join(commit, "conf", f"Model_{label}.json"), "w", encoding="utf-8") as f:
            json.dump({"model": cls, "hidden": 128, "window": 16}, f)
    return commit


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    res = {}
    for fold in ("A", "B"):
        commit = build_routing_commit(fold)
        ev = Evaluator(DATA, commit, device)
        pooled = ev.evaluate(date=VALS[fold])
        daily = {int(d): ev.evaluate(date=d) for d in VALS[fold]}
        extra = {}
        if fold == "A":
            extra["no_date8"] = ev.evaluate(date=[9, 10, 11])
        # latency on first val day
        t0 = time.time()
        ev.evaluate(date=VALS[fold][0])
        lat = time.time() - t0
        res[fold] = {"pooled_official": pooled, "daily": daily, "extra": extra,
                     "latency_sec_day": round(lat, 1),
                     "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
                     "delta_vs_K16_fusion": round(pooled["score"] - FUSION_REF[fold], 4)}
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
        print(f"[{fold}] pooled={pooled} delta_vs_fusion={res[fold]['delta_vs_K16_fusion']} lat={lat:.1f}s", flush=True)
    # full-date-range engineering regression (in-sample, NOT OOS)
    commit_A = os.path.join(OUT, "commit_routing_A")
    ev = Evaluator(DATA, commit_A, device)
    res["full_range_insample_A"] = ev.evaluate(date=None)  # all 20 dates, in-sample diagnostic
    res["full_range_note"] = "训练日期上的评分，仅工程回归诊断，非样本外成绩"
    with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    print("full_range_insample_A:", res["full_range_insample_A"], flush=True)
    print("done", flush=True)
