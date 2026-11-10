# -*- coding: utf-8 -*-
"""TC-37 Gate D0-A: baseline evidence audit (read-only).
Hash core files + v1.zip + y2 checkpoints; recompute UNROUNDED y2 pooled Pearson from frozen
checkpoints (EXP-0016 K16 seed42 y2; EXP-0018 3-seed y2 fusion) via batched inference.
"""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
sys.path.insert(0, os.path.join("project_restart", "src", "models"))
from gru_strong import load_block, make_batch  # noqa: E402
from src.eval.metrics import pooled_pearson_raw  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0024")
os.makedirs(OUT, exist_ok=True)
K = 16


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_models(commit_dir, labels, device="cuda:0"):
    spec = importlib.util.spec_from_file_location("netw_d0", os.path.join(commit_dir, "network.py"),
                                                  submodule_search_locations=[os.path.abspath(commit_dir)])
    netw = importlib.util.module_from_spec(spec)
    sys.modules["netw_d0"] = netw
    spec.loader.exec_module(netw)
    models = []
    for label in labels:
        cfg = json.load(open(os.path.join(commit_dir, "conf", f"Model_{label}.json"), encoding="utf-8"))
        m = getattr(netw, cfg["model"])(cfg).to(device)
        m.load_state_dict(torch.load(os.path.join(commit_dir, "model", f"Model_{label}.pth"),
                                     map_location=device, weights_only=True))
        m.eval()
        models.append(m)
    return models


def predict_y2(commit_dir, dates, device="cuda:0"):
    X, y, tid = load_block(dates)
    models = load_models(commit_dir, [2], device)
    m = models[0]
    outs = []
    with torch.no_grad():
        for s in range(0, len(X), 8192):
            I = np.arange(s, min(s + 8192, len(X)))
            Xg = torch.from_numpy(make_batch(X, I, tid, K)).to(device)
            if hasattr(m, "members"):
                outs.append(torch.stack([mem(Xg) for mem in m.members], 0).mean(0).cpu().numpy())
            else:
                outs.append(m(Xg).cpu().numpy())
    P = np.concatenate(outs)
    c2_unrounded = float(pooled_pearson_raw(P, y[:, 2:3])[0])
    return c2_unrounded, P


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    res = {
        "core_files_sha256": {
            "public.parquet": sha256(DATA),
            "README.md": sha256(os.path.join("project", "project", "README.md")),
            "evaluate.py": sha256(os.path.join("project", "project", "script", "evaluate.py")),
        },
        "tc34_v1_zip_sha256": sha256(os.path.join("project_restart", "releases", "turing_submission_v1.zip")),
        "tc27_protocol": {
            "A": {"inner_fit": [0, 1, 2, 3, 4, 5], "inner_stop": [6, 7], "full_fit": list(range(0, 8)), "outer_val": [8, 9, 10, 11]},
            "B": {"inner_fit": list(range(0, 10)), "inner_stop": [10, 11], "full_fit": list(range(0, 12)), "outer_val": [12, 13, 14, 15]},
        },
        "official_scoring": "Score = round(0.2*raw_c0 + 0.3*raw_c1 + 0.5*raw_c2, 4); corr0/1/2 displayed rounded (R0 confirmed)",
        "y2_baselines": {},
        "y2_checkpoint_sha256": {
            "EXP0016_K16_seed42_y2_A": sha256(os.path.join("project_restart", "experiments", "EXP-0016", "commit_K16_A", "model", "Model_2.pth")),
            "EXP0016_K16_seed42_y2_B": sha256(os.path.join("project_restart", "experiments", "EXP-0016", "commit_K16_B", "model", "Model_2.pth")),
            "EXP0018_ens3_y2_A": sha256(os.path.join("project_restart", "experiments", "EXP-0018", "commit_ens3_K16_A", "model", "Model_2.pth")),
            "EXP0018_ens3_y2_B": sha256(os.path.join("project_restart", "experiments", "EXP-0018", "commit_ens3_K16_B", "model", "Model_2.pth")),
        },
    }
    for name, commit, dates in [
        ("EXP0016_K16_seed42_y2_A", os.path.join("project_restart", "experiments", "EXP-0016", "commit_K16_A"), [8, 9, 10, 11]),
        ("EXP0016_K16_seed42_y2_B", os.path.join("project_restart", "experiments", "EXP-0016", "commit_K16_B"), [12, 13, 14, 15]),
        ("EXP0018_ens3_y2_A", os.path.join("project_restart", "experiments", "EXP-0018", "commit_ens3_K16_A"), [8, 9, 10, 11]),
        ("EXP0018_ens3_y2_B", os.path.join("project_restart", "experiments", "EXP-0018", "commit_ens3_K16_B"), [12, 13, 14, 15]),
    ]:
        c2, P = predict_y2(commit, dates, device)
        res["y2_baselines"][name] = {"c2_unrounded": c2, "c2_rounded": round(c2, 4), "n_pred": int(len(P))}
        print(name, "c2_unrounded =", c2, flush=True)
    with open(os.path.join(OUT, "baseline_evidence.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    print("written baseline_evidence.json")
