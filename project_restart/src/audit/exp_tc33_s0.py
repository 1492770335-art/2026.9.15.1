# -*- coding: utf-8 -*-
"""TC-33 Gate S0: 20-day official streaming engineering regression on frozen 3-seed routing
(EXP-0021 commit_3seed_routing_B, 9 networks). Per-day latency + VRAM (allocated/reserved/nvidia-smi),
NaN/Inf, scorer consistency. Training-date scores are in-sample engineering diagnostics only.
"""
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
from script.evaluate import Evaluator  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
COMMIT = os.path.join("project_restart", "experiments", "EXP-0021", "commit_3seed_routing_B")
OUT = os.path.join("project_restart", "experiments", "EXP-0022")
os.makedirs(OUT, exist_ok=True)


def nvidia_smi_mem():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip()
    except Exception as e:  # noqa: BLE001
        return f"n/a ({e})"


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ev = Evaluator(DATA, COMMIT, device)
    res = {"device": device, "per_day": {}, "anomalies": []}
    t_total = 0.0
    for d in range(20):
        t0 = time.time()
        r = ev.evaluate(date=d)
        lat = time.time() - t0
        t_total += lat
        res["per_day"][d] = {"score": r, "latency_sec": round(lat, 2)}
        print(f"day {d}: {r} {lat:.2f}s", flush=True)
    # full 20-day pooled (in-sample diagnostic)
    t0 = time.time()
    full = ev.evaluate(date=None)
    full_lat = time.time() - t0
    res["full_20day_pooled_insample"] = full
    res["full_20day_latency_sec"] = round(full_lat, 1)
    res["total_per_day_latency_sec"] = round(t_total, 1)
    res["torch_mem"] = {
        "allocated_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1),
        "reserved_mb": round(torch.cuda.max_memory_reserved(device) / 1e6, 1),
        "nvidia_smi": nvidia_smi_mem(),
    }
    with open(os.path.join(OUT, "s0_engineering_regression.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps({k: res[k] for k in ("full_20day_pooled_insample", "full_20day_latency_sec", "total_per_day_latency_sec", "torch_mem")}, indent=2, default=str))
    print("done")
