# -*- coding: utf-8 -*-
"""EXP-0006: seed variance -> noise floor for strong-MLP and strong-GRU on the frozen folds.
- strong-MLP: seeds {42,43,44} x folds {A,B} (6 runs, cheap)
- strong-GRU: seeds {43,44} x fold A (42 already in EXP-0005) to bound GRU noise on the
  most informative fold. Final numbers via official Evaluator.
"""
import json
import os
import sys
import time

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project_restart", "src", "models"))

from mlp_strong import CFG as MLP_CFG, load_block as mlp_load, MLP, train_one as mlp_train_one, build_commit as mlp_commit  # noqa: E402
from gru_strong import CFG as GRU_CFG, load_block as gru_load, GRU, train_one as gru_train_one, build_commit as gru_commit  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0006")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, os.path.join("project", "project"))
from script.evaluate import Evaluator  # noqa: E402


def run_mlp_seed(seed, train_dates, val_dates, device, tag):
    Xtr, ytr = mlp_load(train_dates)
    Xva, yva = mlp_load(val_dates)
    ckpt = os.path.join(OUT, f"ckpt_mlp_{tag}_s{seed}")
    os.makedirs(ckpt, exist_ok=True)
    sds = []
    for label in range(3):
        b, _ = mlp_train_one(Xtr, ytr, label, Xva, yva, device, ckpt, seed)
        sds.append(torch.load(os.path.join(ckpt, f"Model_{label}.pth"), map_location="cpu"))
    commit_dir = os.path.join(OUT, f"commit_mlp_{tag}_s{seed}")
    mlp_commit(sds, commit_dir)
    ev = Evaluator(DATA, commit_dir, "cuda:0")
    return ev.evaluate(date=val_dates)


def run_gru_seed(seed, train_dates, val_dates, device, tag):
    Xtr, ytr, tid_tr = gru_load(train_dates)
    Xva, yva, tid_va = gru_load(val_dates)
    ckpt = os.path.join(OUT, f"ckpt_gru_{tag}_s{seed}")
    os.makedirs(ckpt, exist_ok=True)
    sds = []
    for label in range(3):
        b, _ = gru_train_one(Xtr, ytr, label, Xva, yva, tid_tr, tid_va, device, ckpt, seed)
        sds.append(torch.load(os.path.join(ckpt, f"Model_{label}.pth"), map_location="cpu"))
    commit_dir = os.path.join(OUT, f"commit_gru_{tag}_s{seed}")
    gru_commit(sds, commit_dir)
    ev = Evaluator(DATA, commit_dir, "cuda:0")
    return ev.evaluate(date=val_dates)


if __name__ == "__main__":
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    results = {}
    folds = {"A": (list(range(0, 8)), [8, 9, 10, 11]), "B": (list(range(0, 12)), [12, 13, 14, 15])}
    for tag, (tr, va) in folds.items():
        for seed in (42, 43, 44):
            t0 = time.time()
            r = run_mlp_seed(seed, tr, va, device, tag)
            results[f"mlp_{tag}_s{seed}"] = {"seed": seed, "fold": tag, "score": r, "elapsed_sec": round(time.time() - t0, 1)}
            print(json.dumps(results[f"mlp_{tag}_s{seed}"]), flush=True)
    # GRU fold A seeds 43, 44 (42 from EXP-0005)
    trA, vaA = folds["A"]
    for seed in (43, 44):
        t0 = time.time()
        r = run_gru_seed(seed, trA, vaA, device, "A")
        results[f"gru_A_s{seed}"] = {"seed": seed, "fold": "A", "score": r, "elapsed_sec": round(time.time() - t0, 1)}
        print(json.dumps(results[f"gru_A_s{seed}"]), flush=True)
    with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    print("done", flush=True)
