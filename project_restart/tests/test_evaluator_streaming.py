# -*- coding: utf-8 -*-
"""Official streaming Evaluator reconciliation tests (needs torch+polars).
Run: python tests/test_evaluator_streaming.py
Covers: pooled Pearson equality with independent implementation, per-day reset
frequency, reset-then-empty-history on day boundary, stateless MLP determinism.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJ = os.path.join(ROOT, "..", "project", "project")  # official package
sys.path.insert(0, PROJ)

from src.eval.metrics import pooled_pearson_raw, official_score  # noqa: E402

FAIL = []


def check(name, cond, detail=""):
    print(("[PASS] " if cond else "[FAIL] ") + name + ("" if cond else f" {detail}"))
    if not cond:
        FAIL.append(name)


def make_instrumented_commit(src_commit: str, reset_log: str, kind: str) -> str:
    tmp = tempfile.mkdtemp(prefix="instr_commit_")
    for sub in ["conf", "model"]:
        shutil.copytree(os.path.join(src_commit, sub), os.path.join(tmp, sub))
    base_src = open(os.path.join(src_commit, "base.py"), encoding="utf-8").read()
    with open(os.path.join(tmp, "base.py"), "w", encoding="utf-8") as f:
        f.write(base_src)
    if kind == "mlp":
        net = (
            "from torch import nn\n"
            "from .base import BaseModel\n"
            "import os\n"
            "LOG = os.path.join(os.path.dirname(__file__), 'reset_log.txt')\n"
            "class MLP(BaseModel):\n"
            "    def __init__(self, config):\n"
            "        super().__init__(config)\n"
            "        hidden = config['hidden']\n"
            "        self.network = nn.Sequential(nn.Linear(100, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))\n"
            "    def forward(self, x):\n"
            "        return self.network(x)\n"
            "    def reset(self):\n"
            "        with open(LOG, 'a') as f:\n"
            "            f.write('r\\n')\n"
            "    def predict(self, x):\n"
            "        return self.forward(x)\n"
        )
    else:  # gru
        net = (
            "import torch\n"
            "from torch import nn\n"
            "from torch.nn import functional as F\n"
            "from .base import BaseModel\n"
            "import os\n"
            "LOG = os.path.join(os.path.dirname(__file__), 'reset_log.txt')\n"
            "HIST = os.path.join(os.path.dirname(__file__), 'hist_at_tick0.txt')\n"
            "class GRU(BaseModel):\n"
            "    def __init__(self, config):\n"
            "        super().__init__(config)\n"
            "        hidden = config['hidden']\n"
            "        self.gru = nn.GRU(100, hidden, batch_first=True)\n"
            "        self.output = nn.Linear(hidden, 1)\n"
            "        self.window = config.get('window', 100)\n"
            "        self.history = None\n"
            "        self.first = True\n"
            "    def forward(self, x):\n"
            "        h, _ = self.gru(x)\n"
            "        return self.output(h[:, -1])\n"
            "    def reset(self):\n"
            "        with open(LOG, 'a') as f:\n"
            "            f.write('r\\n')\n"
            "        self.first = True\n"
            "        self.history = None\n"
            "    def predict(self, x):\n"
            "        if self.first:\n"
            "            self.first = False\n"
            "            with open(HIST, 'a') as f:\n"
            "                f.write('empty=%s\\n' % (self.history is None))\n"
            "        current = x[:, None].detach()\n"
            "        self.history = current if self.history is None else torch.cat([self.history, current], dim=1)\n"
            "        self.history = self.history[:, -self.window:]\n"
            "        history = F.pad(self.history, (0, 0, self.window - self.history.shape[1], 0))\n"
            "        return self.forward(history)\n"
        )
    with open(os.path.join(tmp, "network.py"), "w", encoding="utf-8") as f:
        f.write(net)
    return tmp


def mini_evaluator(parquet_path, commit_path, device="cpu", date=None):
    """Independent reimplementation of script/evaluate.py L33-L67 (kept verbatim semantics)."""
    from script.evaluate import Evaluator  # official loader for models only

    data = pl.scan_parquet(parquet_path)
    if date is None:
        dates = data.select(pl.col("date").unique().sort()).collect()["date"].to_list()
    else:
        dates = [date] if isinstance(date, int) else sorted(set(date))
    models = Evaluator(parquet_path, commit_path, device).models
    sums = np.zeros((6, 3), dtype=np.float64)
    with torch.no_grad():
        for day in dates:
            frame = data.filter(pl.col("date") == day).sort(["time", "code"]).collect()
            features = sorted([n for n in frame.columns if n.startswith("f")], key=lambda n: int(n[1:]))
            x = frame.select(features).to_numpy().astype(np.float32).reshape(-1, 50, 100)
            y = frame.select(["y0", "y1", "y2"]).to_numpy().astype(np.float64)
            preds = np.empty((len(x), 50, 3), dtype=np.float64)
            for m in models:
                m.reset()
            for tick, cur in enumerate(x):
                t = torch.from_numpy(cur).to(device)
                for label, m in enumerate(models):
                    preds[tick, :, label] = m.predict(t.clone())[:, 0].cpu().numpy()
            p = preds.reshape(-1, 3)
            sums += np.stack((np.full(3, len(y)), p.sum(0), y.sum(0), (p * p).sum(0), (y * y).sum(0), (p * y).sum(0)))
    n, p, y, pp, yy, py = sums
    denom = np.sqrt(np.maximum(pp - p * p / n, 0) * np.maximum(yy - y * y / n, 0))
    corr = (py - p * y / n) / denom
    return {
        "corr0": round(float(corr[0]), 4), "corr1": round(float(corr[1]), 4),
        "corr2": round(float(corr[2]), 4), "score": round(float(corr @ np.array([0.2, 0.3, 0.5])), 4),
    }


def main():
    data = os.path.join(PROJ, "data", "public.parquet")
    sys.path.insert(0, os.path.join(PROJ, "script"))
    from script.evaluate import Evaluator

    # --- Test B1: MLP streaming vs mini implementation, dates 8,9 ---
    tmp_mlp = make_instrumented_commit(os.path.join(PROJ, "commit_v0"), "reset_log.txt", "mlp")
    ev = Evaluator(data, tmp_mlp, "cpu")
    got = ev.evaluate(date=[8, 9])
    mine = mini_evaluator(data, tmp_mlp, "cpu", date=[8, 9])
    check("MLP official==mini (dates 8-9)", got == mine, f"official={got} mini={mine}")
    with open(os.path.join(tmp_mlp, "reset_log.txt")) as f:
        resets = f.read().count("r\n")
    # 2 Evaluator instances (official + mini) x 3 models x 2 days = 12; i.e. once per model per day per instance
    check("MLP reset once per model per day (12 = 2 inst x 3 models x 2 days)", resets == 12, f"resets={resets}")

    # --- Test B2: GRU streaming vs mini implementation + day-boundary reset semantics ---
    tmp_gru = make_instrumented_commit(os.path.join(PROJ, "commit_v1"), "reset_log.txt", "gru")
    got = Evaluator(data, tmp_gru, "cpu").evaluate(date=[8, 9])
    mine = mini_evaluator(data, tmp_gru, "cpu", date=[8, 9])
    check("GRU official==mini (dates 8-9)", got == mine, f"official={got} mini={mine}")
    with open(os.path.join(tmp_gru, "reset_log.txt")) as f:
        resets = f.read().count("r\n")
    check("GRU reset once per model per day (12 = 2 inst x 3 models x 2 days)", resets == 12, f"resets={resets}")
    with open(os.path.join(tmp_gru, "hist_at_tick0.txt")) as f:
        lines = [l.strip() for l in f if l.strip()]
    check("GRU history empty at first tick after each reset", all(l == "empty=True" for l in lines), str(lines))

    # --- Test B3: score math on official outputs equals raw recomputation ---
    p = np.random.default_rng(1).normal(size=(100, 3))
    y = np.random.default_rng(2).normal(size=(100, 3))
    o = official_score(p, y)
    c = pooled_pearson_raw(p, y)
    check("metrics.py mirrors evaluate.py rounding", o["corr0"] == round(float(c[0]), 4) and abs(o["raw_score"] - float(c @ np.array([0.2, 0.3, 0.5]))) < 1e-12)

    # --- Test B4: NaN prediction triggers official error path (commit with NaN MLP) ---
    tmp_nan = make_instrumented_commit(os.path.join(PROJ, "commit_v0"), "reset_log.txt", "mlp")
    net_src = open(os.path.join(tmp_nan, "network.py"), encoding="utf-8").read().replace(
        "return self.forward(x)", "import torch\n        z = self.forward(x); z[0, 0] = float('nan'); return z")
    with open(os.path.join(tmp_nan, "network.py"), "w", encoding="utf-8") as f:
        f.write(net_src)
    try:
        Evaluator(data, tmp_nan, "cpu").evaluate(date=8)
        check("NaN prediction raises ValueError", False)
    except ValueError as e:
        check("NaN prediction raises ValueError", "NaN/Inf" in str(e), str(e))

    print()
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)
    print("ALL PASS")


if __name__ == "__main__":
    main()
