# -*- coding: utf-8 -*-
"""TC-34 Gate S2 F2: assemble standalone submission package + official 20-day engineering acceptance.
Builds releases/turing_submission_v1/{base.py, network.py, conf/Model_0..2.json, model/Model_0..2.pth, description.md},
zips it, writes SHA manifest, then runs the UNMODIFIED official Evaluator on all 20 dates from an
independent copy. 20-day score is IN_SAMPLE_ENGINEERING_DIAGNOSTIC only.
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join("project", "project"))
sys.path.insert(0, os.path.join("project", "project", "script"))
from script.evaluate import Evaluator  # noqa: E402

EXP23 = os.path.join("project_restart", "experiments", "EXP-0023")
RELEASE = os.path.join("project_restart", "releases", "turing_submission_v1")
ZIP = os.path.join("project_restart", "releases", "turing_submission_v1.zip")
DATA = os.path.join("project", "project", "data", "public.parquet")

NET = (
    "import torch\n"
    "from torch import nn\n"
    "from torch.nn import functional as F\n"
    "from .base import BaseModel\n"
    "class _GRU100(nn.Module):\n"
    "    def __init__(self, hidden):\n"
    "        super().__init__()\n"
    "        self.gru = nn.GRU(100, hidden, batch_first=True)\n"
    "        self.output = nn.Linear(hidden, 1)\n"
    "    def forward(self, x):\n"
    "        h, _ = self.gru(x)\n"
    "        return self.output(h[:, -1])\n"
    "class _GRU300(nn.Module):\n"
    "    def __init__(self, hidden):\n"
    "        super().__init__()\n"
    "        self.gru = nn.GRU(300, hidden, batch_first=True)\n"
    "        self.output = nn.Linear(hidden, 1)\n"
    "    def forward(self, x):\n"
    "        h, _ = self.gru(x)\n"
    "        return self.output(h[:, -1])\n"
    "class MarketEnsemble(BaseModel):\n"
    "    def __init__(self, config):\n"
    "        super().__init__(config)\n"
    "        hidden = config['hidden']\n"
    "        self.window = config.get('window', 16)\n"
    "        self.members = nn.ModuleList([_GRU300(hidden) for _ in range(3)])\n"
    "        self.hist = [None, None, None]\n"
    "    def reset(self):\n"
    "        self.hist = [None, None, None]\n"
    "    def predict(self, x):\n"
    "        m = x.mean(0, keepdim=True)\n"
    "        s = x.std(0, keepdim=True)\n"
    "        xm = torch.cat([x, m.expand_as(x), s.expand_as(x)], dim=1)\n"
    "        cur = xm[:, None].detach()\n"
    "        outs = []\n"
    "        for k, mem in enumerate(self.members):\n"
    "            self.hist[k] = cur if self.hist[k] is None else torch.cat([self.hist[k], cur], dim=1)\n"
    "            self.hist[k] = self.hist[k][:, -self.window:]\n"
    "            h = F.pad(self.hist[k], (0, 0, self.window - self.hist[k].shape[1], 0))\n"
    "            outs.append(mem(h))\n"
    "        return torch.stack(outs, 0).mean(0)\n"
    "class K16Ensemble(BaseModel):\n"
    "    def __init__(self, config):\n"
    "        super().__init__(config)\n"
    "        hidden = config['hidden']\n"
    "        self.window = config.get('window', 16)\n"
    "        self.members = nn.ModuleList([_GRU100(hidden) for _ in range(3)])\n"
    "        self.hist = [None, None, None]\n"
    "    def reset(self):\n"
    "        self.hist = [None, None, None]\n"
    "    def predict(self, x):\n"
    "        cur = x[:, None].detach()\n"
    "        outs = []\n"
    "        for k, mem in enumerate(self.members):\n"
    "            self.hist[k] = cur if self.hist[k] is None else torch.cat([self.hist[k], cur], dim=1)\n"
    "            self.hist[k] = self.hist[k][:, -self.window:]\n"
    "            h = F.pad(self.hist[k], (0, 0, self.window - self.hist[k].shape[1], 0))\n"
    "            outs.append(mem(h))\n"
    "        return torch.stack(outs, 0).mean(0)\n"
)

DESC = """# 图灵杯高频预测提交说明（三种子目标路由）

## 模型结构
三个目标使用不同网络（README 允许三目标不同架构）：
- y0：MarketGRU K16（GRU128，隐藏 128；输入 300 = 100 个股特征 + 当前 tick 横截面 mean/std），seed42/43/44 三个模型预测 1:1:1 等权平均。
- y1：同 y0（MarketGRU K16，三 seed 等权）。
- y2：非 Market K16-GRU（GRU128，隐藏 128；输入 100 原始特征），seed42/43/44 三个模型预测 1:1:1 等权平均。

## 训练
- 数据：公开 date0–19（全 20 日）；每个目标独立训练；y0 固定 4 epoch、y1 固定 2 epoch、y2 固定 2 epoch（epoch 数由 0–15 内部验证预注册，不在 full20 重搜）。
- 优化：AdamW lr=1e-4、weight_decay=1e-4；MSE 损失；batch=1024；梯度裁剪 1.0；seed=42/43/44。
- 因果性：每股票同日因果窗口 K=16，左零填充，日初 reset；市场统计仅用当前 tick 50 股特征（mean/std，ddof=1），无未来/标签信息。

## 加载
- Model_0/1 的 .pth 各含 members.0/1/2 三个 seed 的 MarketGRU 权重；Model_2 含三个 seed 的 K16-GRU 权重。
- 推理仅需 torch（网络类仅用 torch.nn/torch.nn.functional）；输入按官方接口 float32 [50,100]，输出 [50,1]。

## 依赖
torch（参见 requirements.txt：numpy/polars/torch/tqdm）。
"""


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def build_release():
    if os.path.exists(RELEASE):
        shutil.rmtree(RELEASE)
    os.makedirs(os.path.join(RELEASE, "conf"))
    os.makedirs(os.path.join(RELEASE, "model"))
    shutil.copy(os.path.join("project", "project", "script", "base.py"), os.path.join(RELEASE, "base.py"))
    with open(os.path.join(RELEASE, "network.py"), "w", encoding="utf-8") as f:
        f.write(NET)
    with open(os.path.join(RELEASE, "description.md"), "w", encoding="utf-8") as f:
        f.write(DESC)
    for label, cls, tagprefix in ((0, "MarketEnsemble", "market_full"), (1, "MarketEnsemble", "market_full"), (2, "K16Ensemble", "k16_full")):
        ens_sd = {}
        for i, seed in enumerate((42, 43, 44)):
            p = os.path.join(EXP23, f"{tagprefix}_s{seed}_y{label}.pth")
            state = torch.load(p, map_location="cpu", weights_only=True)
            for k, v in state.items():
                ens_sd[f"members.{i}.{k}"] = v
        torch.save(ens_sd, os.path.join(RELEASE, "model", f"Model_{label}.pth"))
        with open(os.path.join(RELEASE, "conf", f"Model_{label}.json"), "w", encoding="utf-8") as f:
            json.dump({"model": cls, "hidden": 128, "window": 16}, f)
    return RELEASE


def nvidia_smi():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


def ram_peak_mb():
    try:
        import psutil
        return round(psutil.Process().memory_info().peak_wset / 1e6, 1) if hasattr(psutil.Process().memory_info(), "peak_wset") else "n/a"
    except Exception:  # noqa: BLE001
        return "n/a"


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    build_release()
    # SHA manifest
    manifest = {}
    for root, _, files in os.walk(RELEASE):
        for fn in sorted(files):
            p = os.path.join(root, fn)
            manifest[os.path.relpath(p, os.path.dirname(RELEASE))] = sha256(p)
    # zip
    if os.path.exists(ZIP):
        os.remove(ZIP)
    shutil.make_archive(ZIP[:-4], "zip", os.path.dirname(RELEASE), os.path.basename(RELEASE))
    manifest["_zip_sha256"] = sha256(ZIP)
    with open(os.path.join(RELEASE, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    # --- official acceptance from an independent copy ---
    tmp = os.path.join("project_restart", "releases", "_accept_check")
    if os.path.exists(tmp):
        shutil.rmtree(tmp)
    shutil.copytree(RELEASE, tmp)
    ev = Evaluator(DATA, tmp, device)
    per_day = {}
    t_total = 0.0
    for d in range(20):
        t0 = time.time()
        r = ev.evaluate(date=d)
        lat = time.time() - t0
        t_total += lat
        per_day[d] = {"score": r, "latency_sec": round(lat, 2)}
    full = ev.evaluate(date=None)
    acc = {
        "full_20day_insample": full,
        "per_day": per_day,
        "total_runtime_sec": round(t_total, 1),
        "torch_allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
        "torch_reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1e6, 1),
        "nvidia_smi": nvidia_smi(),
        "ram_peak_mb": ram_peak_mb(),
        "note": "20 日已参与训练，full_20day_insample 为 IN_SAMPLE_ENGINEERING_DIAGNOSTIC，非 OOS",
    }
    with open(os.path.join(EXP23, "s2_acceptance.json"), "w", encoding="utf-8") as f:
        json.dump(acc, f, indent=2, default=str)
    shutil.rmtree(tmp)
    print(json.dumps({"full_20day_insample": full, "total_runtime_sec": acc["total_runtime_sec"],
                      "torch_allocated_mb": acc["torch_allocated_mb"], "torch_reserved_mb": acc["torch_reserved_mb"],
                      "nvidia_smi": acc["nvidia_smi"], "ram_peak_mb": acc["ram_peak_mb"]}, indent=2, default=str))
    print("release:", RELEASE)
    print("zip:", ZIP, "sha:", manifest["_zip_sha256"])
    print("done")
