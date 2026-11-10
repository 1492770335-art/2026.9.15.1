# -*- coding: utf-8 -*-
"""Build a commit/ dir for the Ridge baseline and verify via the OFFICIAL Evaluator.
Run: python src/models/ridge_commit.py
"""
import json
import os
import sys
import shutil

import numpy as np
import polars as pl
import torch

sys.path.insert(0, os.path.join("project_restart"))
from src.models.ridge import load_block, fit_ridge, predict_ridge  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
COMMIT = os.path.join("project_restart", "experiments", "EXP-0001", "commit_ridge_foldA_lam100")
if os.path.exists(COMMIT):
    shutil.rmtree(COMMIT)
os.makedirs(os.path.join(COMMIT, "conf"))
os.makedirs(os.path.join(COMMIT, "model"))

base_py = open(os.path.join("project", "project", "script", "base.py"), encoding="utf-8").read()
net_py = """from torch import nn
import torch
from .base import BaseModel

class Ridge(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.W = nn.Parameter(torch.zeros(101, 1))

    def reset(self):
        pass

    def predict(self, x):
        ones = torch.ones(x.shape[0], 1, device=x.device, dtype=x.dtype)
        return torch.cat([x, ones], dim=1) @ self.W
"""
with open(os.path.join(COMMIT, "base.py"), "w", encoding="utf-8") as f:
    f.write(base_py)
with open(os.path.join(COMMIT, "network.py"), "w", encoding="utf-8") as f:
    f.write(net_py)

train_dates = list(range(0, 8))
Xtr, ytr, _ = load_block(DATA, train_dates)
W = fit_ridge(Xtr, ytr, 100.0)  # (101, 3) float64

for label in range(3):
    cfg = {"model": "Ridge", "label": label}
    with open(os.path.join(COMMIT, "conf", f"Model_{label}.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    sd = {"W": torch.from_numpy(W[:, label:label + 1].copy())}
    torch.save(sd, os.path.join(COMMIT, "model", f"Model_{label}.pth"))

# official evaluator check (dates 8-11 == fold A val)
sys.path.insert(0, os.path.join("project", "project"))
from script.evaluate import Evaluator  # noqa: E402

ev = Evaluator(DATA, COMMIT, "cpu")
result = ev.evaluate(date=[8, 9, 10, 11])
print("OFFICIAL evaluator, fold A val 8-11:", json.dumps(result))
assert abs(result["score"] - 0.1929) <= 0.0001, f"mismatch: {result}"
print("matches closed-form score 0.1929 (lam=100) ✓")
