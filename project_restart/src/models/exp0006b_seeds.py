# -*- coding: utf-8 -*-
"""EXP-0006b (受限恢复): strong-GRU seed 43/44 on Fold A and Fold B, identical recipe to EXP-0005
except seed. NO overwrite policy:
- reuses existing EXP-0006/ckpt_gru_A_s43/Model_0.pth (pause-era, verified loadable) without retraining;
- trains missing targets into that same dir (Model_1/Model_2 are new files);
- all other ckpt/commit/results written to NEW paths.
Runs ONLY this experiment (no queue).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
sys.path.insert(0, os.path.join("project_restart", "src", "models"))
from script.evaluate import Evaluator  # noqa: E402
from gru_strong import load_block, train_one, build_commit, CFG  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0006")
RESULT = os.path.join(OUT, "results_raw_gru_seeds.json")

TASKS = [
    {"fold": "A", "train": list(range(0, 8)), "val": [8, 9, 10, 11], "seed": 43,
     "ckpt": os.path.join(OUT, "ckpt_gru_A_s43"), "commit": os.path.join(OUT, "commit_gru_A_s43")},
    {"fold": "A", "train": list(range(0, 8)), "val": [8, 9, 10, 11], "seed": 44,
     "ckpt": os.path.join(OUT, "ckpt_gru_A_s44"), "commit": os.path.join(OUT, "commit_gru_A_s44")},
    {"fold": "B", "train": list(range(0, 12)), "val": [12, 13, 14, 15], "seed": 43,
     "ckpt": os.path.join(OUT, "ckpt_gru_B_s43"), "commit": os.path.join(OUT, "commit_gru_B_s43")},
    {"fold": "B", "train": list(range(0, 12)), "val": [12, 13, 14, 15], "seed": 44,
     "ckpt": os.path.join(OUT, "ckpt_gru_B_s44"), "commit": os.path.join(OUT, "commit_gru_B_s44")},
]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_results():
    if os.path.exists(RESULT):
        with open(RESULT, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_results(res):
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)


def run_task(task, device):
    t0 = time.time()
    tag = f"gru_{task['fold']}_s{task['seed']}"
    os.makedirs(task["ckpt"], exist_ok=True)
    Xtr, ytr, tid_tr = load_block(task["train"])
    Xva, yva, tid_va = load_block(task["val"])
    sds, notes = [], {}
    for label in range(3):
        ckpt_file = os.path.join(task["ckpt"], f"Model_{label}.pth")
        if os.path.exists(ckpt_file):
            sd = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            notes[f"Model_{label}"] = {"reused": True, "sha256": sha256(ckpt_file)}
            sds.append(sd)
            print(f"[reuse] {ckpt_file} {notes[f'Model_{label}']['sha256'][:16]}", flush=True)
        else:
            b, h = train_one(Xtr, ytr, label, Xva, yva, tid_tr, tid_va, device, task["ckpt"], task["seed"])
            sd = torch.load(ckpt_file, map_location="cpu", weights_only=True)
            notes[f"Model_{label}"] = {"reused": False, "best_epoch": b.get("epoch"),
                                       "sha256": sha256(ckpt_file)}
            sds.append(sd)
            print(f"[trained] {ckpt_file} best_epoch={b.get('epoch')}", flush=True)
    build_commit(sds, task["commit"])
    ev = Evaluator(DATA, task["commit"], "cuda:0")
    pooled = ev.evaluate(date=task["val"])
    daily = {d: ev.evaluate(date=d) for d in task["val"]}
    # date-8 removal diagnostic for fold A (val = 8-11): pooled over 9-11
    extra = {}
    if task["fold"] == "A":
        extra["pooled_no_date8"] = ev.evaluate(date=[9, 10, 11])
    rec = {"tag": tag, "fold": task["fold"], "seed": task["seed"], "train": task["train"], "val": task["val"],
           "pooled": pooled, "daily": daily, "extra": extra, "notes": notes,
           "elapsed_sec": round(time.time() - t0, 1),
           "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1)}
    torch.cuda.reset_peak_memory_stats(device)
    return tag, rec


if __name__ == "__main__":
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device, "CFG:", {k: CFG[k] for k in ("hidden", "window", "lr", "epochs", "batch_size", "seed")}, flush=True)
    results = load_results()
    for task in TASKS:
        tag, rec = run_task(task, device)
        results[tag] = rec
        save_results(results)
        print(json.dumps({k: rec[k] for k in ("fold", "seed", "pooled", "daily", "extra", "elapsed_sec", "peak_vram_mb")}, indent=2), flush=True)
    print("done", flush=True)
