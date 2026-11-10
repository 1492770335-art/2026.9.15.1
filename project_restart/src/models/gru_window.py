# -*- coding: utf-8 -*-
"""EXP-0007: GRU window ablation K in {16, 64} on both folds (K=100 reference = EXP-0005).
Single variable: window length only; recipe identical to strong GRU (gru_strong.py).
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project_restart", "src", "models"))

import gru_strong  # noqa: E402

OUT = os.path.join("project_restart", "experiments", "EXP-0007")
os.makedirs(OUT, exist_ok=True)
gru_strong.OUT = OUT

if __name__ == "__main__":
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tasks = {
        "foldA": (list(range(0, 8)), [8, 9, 10, 11]),
        "foldB": (list(range(0, 12)), [12, 13, 14, 15]),
    }
    results = {}
    for K in (16, 64):
        gru_strong.CFG["window"] = K
        for name, (tr, va) in tasks.items():
            tag = f"{name}_K{K}"
            t0 = time.time()
            print("running", tag, flush=True)
            r = gru_strong.run_task(tag, tr, va, device)
            r["elapsed_sec"] = round(time.time() - t0, 1)
            results[tag] = r
            print(json.dumps({k: r[k] for k in ("pooled", "daily", "best", "elapsed_sec", "peak_vram_mb")}, indent=2), flush=True)
    with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("done", flush=True)
