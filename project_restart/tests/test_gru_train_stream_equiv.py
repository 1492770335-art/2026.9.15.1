# -*- coding: utf-8 -*-
"""Training/streaming equivalence test for GRU: batched make_batch predictions (training semantics)
must reproduce the official streaming Evaluator scores exactly for a trained commit.
Run after GPU queue: python tests/test_gru_train_stream_equiv.py
"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
from script.evaluate import Evaluator  # noqa: E402
from src.eval.metrics import official_score  # noqa: E402

sys.path.insert(0, os.path.join("project_restart", "src", "models"))
from gru_strong import load_block, make_batch, GRU, CFG  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
COMMIT = os.path.join("project_restart", "experiments", "EXP-0003", "commit_gru_foldA")
EXPECTED = 0.2012  # official pooled score dates 8-11

if __name__ == "__main__":
    torch.set_num_threads(8)
    dev = "cpu"
    # load models
    import importlib.util
    spec = importlib.util.spec_from_file_location("netw", os.path.join(COMMIT, "network.py"),
                                                  submodule_search_locations=[os.path.abspath(COMMIT)])
    netw = importlib.util.module_from_spec(spec)
    sys.modules["netw"] = netw
    spec.loader.exec_module(netw)
    models = []
    for label in range(3):
        cfg = json.load(open(os.path.join(COMMIT, "conf", f"Model_{label}.json")))
        m = getattr(netw, cfg["model"])(cfg).to(dev)
        m.load_state_dict(torch.load(os.path.join(COMMIT, "model", f"Model_{label}.pth"), map_location=dev, weights_only=True))
        m.eval()
        models.append(m)

    all_p, all_y = [], []
    for d in [8, 9, 10, 11]:
        X, y, tid = load_block([d])
        chunk = 4096
        # per-model batched predictions (GRU full forward via model(Xg))
        preds = []
        for m in models:
            o = []
            with torch.no_grad():
                for s in range(0, len(X), chunk):
                    I = np.arange(s, min(s + chunk, len(X)))
                    Xg = torch.from_numpy(make_batch(X, I, tid, CFG["window"])).to(dev)
                    o.append(m(Xg).numpy())
            preds.append(np.concatenate(o))
        P = np.stack(preds, axis=1).reshape(-1, 3)
        all_p.append(P)
        all_y.append(y)
    P = np.concatenate(all_p)
    Y = np.concatenate(all_y)
    o = official_score(P, Y)
    print("batched-pred official-math score:", json.dumps(o))
    ok = abs(o["score"] - EXPECTED) < 0.0002
    print("MATCH" if ok else "MISMATCH vs streaming official 0.2012")
    sys.exit(0 if ok else 1)
