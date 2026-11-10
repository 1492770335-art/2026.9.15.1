# -*- coding: utf-8 -*-
"""EXP-0015 (TC-27): internal early-stop protocol implementation + new-protocol K100 strong-GRU seed42.
Protocol (pre-registered EARLYSTOP_AUDIT.md):
  fold A: inner_train 0-5, inner_stop 6-7, full_train 0-7, outer_val 8-11
  fold B: inner_train 0-9, inner_stop 10-11, full_train 0-11, outer_val 12-15
Per target y0/y1/y2:
  phase1: train on inner_train; every 2 epochs evaluate own-target pooled Pearson on inner_stop;
          select best epoch E (patience 4, cap 30). OUTER VAL NEVER USED for selection.
  phase2: retrain on full_train for EXACTLY E epochs (same seed 42, same rng(seed+epoch) shuffles);
          save checkpoint; evaluate ONCE on outer_val via official Evaluator (pooled + daily).
Structure/hyperparams identical to EXP-0005 (GRU 128/window100/lr1e-4/wd1e-4/clip1.0/batch1024).
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
sys.path.insert(0, os.path.join("project_restart", "src", "models"))
from script.evaluate import Evaluator  # noqa: E402
from gru_strong import load_block, make_batch, GRU, predict_batched, build_commit, CFG  # noqa: E402
from src.eval.metrics import pooled_pearson_raw  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0015")
os.makedirs(OUT, exist_ok=True)

CAP = 30
PATIENCE = 4
EVAL_EVERY = 2
SEED = 42

FOLDS = {
    "A": {"inner_train": [0, 1, 2, 3, 4, 5], "inner_stop": [6, 7],
          "full_train": list(range(0, 8)), "outer_val": [8, 9, 10, 11]},
    "B": {"inner_train": list(range(0, 10)), "inner_stop": [10, 11],
          "full_train": list(range(0, 12)), "outer_val": [12, 13, 14, 15]},
}


def train_epoch_fn(model, X, Y, tid, order, opt, device):
    model.train()
    total, cnt = 0.0, 0
    for s in range(0, len(order), CFG["batch_size"]):
        I = order[s:s + CFG["batch_size"]]
        Xg = torch.from_numpy(make_batch(X, I, tid, CFG["window"])).to(device)
        loss = F.mse_loss(model(Xg), Y[I])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["clip"])
        opt.step()
        total += loss.detach().item() * len(I)
        cnt += len(I)
    return total / cnt


def select_epoch(X, y, tid, label, Xst, yst, tid_st, device):
    torch.manual_seed(SEED)
    model = GRU(CFG["hidden"]).to(device)
    Y = torch.from_numpy(y[:, label:label + 1].astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    n = len(X)
    best = {"epoch": -1, "c": -2.0}
    traj = []
    for epoch in range(CAP):
        rng = np.random.default_rng(SEED + epoch)
        order = rng.permutation(n)
        train_loss = train_epoch_fn(model, X, Y, tid, order, opt, device)
        if (epoch + 1) % EVAL_EVERY == 0:
            p = predict_batched(model, Xst, tid_st, device)
            c = pooled_pearson_raw(p, yst[:, label:label + 1])[0]
            traj.append({"epoch": epoch + 1, "train_loss": train_loss, "inner_stop_corr": float(c)})
            if c > best["c"]:
                best = {"epoch": epoch + 1, "c": float(c)}
            if epoch + 1 - best["epoch"] >= PATIENCE:
                break
    return best["epoch"], traj


def retrain_full(X, y, tid, label, E, device, out_path):
    torch.manual_seed(SEED)
    model = GRU(CFG["hidden"]).to(device)
    Y = torch.from_numpy(y[:, label:label + 1].astype(np.float32)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    n = len(X)
    batches_per_epoch = math.ceil(n / CFG["batch_size"])
    losses = []
    for epoch in range(E):
        rng = np.random.default_rng(SEED + epoch)
        order = rng.permutation(n)
        losses.append(train_epoch_fn(model, X, Y, tid, order, opt, device))
    torch.save(model.state_dict(), out_path)
    return batches_per_epoch * E, losses


if __name__ == "__main__":
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device, "CFG:", {k: CFG[k] for k in ("hidden", "window", "lr", "epochs", "batch_size", "seed")}, flush=True)
    results = {}
    for fold, spec in FOLDS.items():
        t0 = time.time()
        Xi, yi, tii = load_block(spec["inner_train"])
        Xs, ys, tis = load_block(spec["inner_stop"])
        Xf, yf, tif = load_block(spec["full_train"])
        ckpt = os.path.join(OUT, f"ckpt_np_{fold}")
        os.makedirs(ckpt, exist_ok=True)
        info = {}
        sds = []
        for label in range(3):
            E, traj = select_epoch(Xi, yi, tii, label, Xs, ys, tis, device)
            out_path = os.path.join(ckpt, f"Model_{label}.pth")
            steps, losses = retrain_full(Xf, yf, tif, label, E, device, out_path)
            sds.append(torch.load(out_path, map_location="cpu", weights_only=True))
            info[f"y{label}"] = {"E": E, "inner_traj": traj, "full_train_losses": losses,
                                 "inner_batches_per_epoch": math.ceil(len(Xi) / CFG["batch_size"]),
                                 "full_batches_per_epoch": math.ceil(len(Xf) / CFG["batch_size"]),
                                 "total_opt_steps_full": steps}
            print(f"[{fold} y{label}] E={E} steps={steps}", flush=True)
        commit_dir = os.path.join(OUT, f"commit_k100_np_{fold}")
        build_commit(sds, commit_dir)
        ev = Evaluator(DATA, commit_dir, device)
        pooled = ev.evaluate(date=spec["outer_val"])
        daily = {int(d): ev.evaluate(date=d) for d in spec["outer_val"]}
        # batched-vs-streaming cross-check on first outer val date
        Xv, yv, tiv = load_block([spec["outer_val"][0]])
        import importlib.util
        spec_mod = importlib.util.spec_from_file_location("netw2", os.path.join(commit_dir, "network.py"),
                                                          submodule_search_locations=[os.path.abspath(commit_dir)])
        netw2 = importlib.util.module_from_spec(spec_mod)
        sys.modules["netw2"] = netw2
        spec_mod.loader.exec_module(netw2)
        models = []
        for label in range(3):
            cfg = json.load(open(os.path.join(commit_dir, "conf", f"Model_{label}.json"), encoding="utf-8"))
            m = getattr(netw2, cfg["model"])(cfg).to(device)
            m.load_state_dict(torch.load(os.path.join(commit_dir, "model", f"Model_{label}.pth"),
                                         map_location=device, weights_only=True))
            m.eval()
            models.append(m)
        P = []
        with torch.no_grad():
            for m in models:
                o = []
                for s in range(0, len(Xv), 4096):
                    I = np.arange(s, min(s + 4096, len(Xv)))
                    Xg = torch.from_numpy(make_batch(Xv, I, tiv, CFG["window"])).to(device)
                    o.append(m(Xg).cpu().numpy())
                P.append(np.concatenate(o))
        P = np.stack(P, axis=1).reshape(-1, 3)
        from src.eval.metrics import official_score
        batched = official_score(P, yv)
        rec = {"fold": fold, "inner_train": spec["inner_train"], "inner_stop": spec["inner_stop"],
               "full_train": spec["full_train"], "outer_val": spec["outer_val"],
               "per_target": info, "pooled_official": pooled, "daily": daily,
               "batched_check_first_val_day": batched,
               "elapsed_sec": round(time.time() - t0, 1),
               "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1)}
        results[fold] = rec
        with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        print(json.dumps({k: rec[k] for k in ("per_target", "pooled_official", "daily", "batched_check_first_val_day", "elapsed_sec", "peak_vram_mb")}, indent=2, default=str), flush=True)
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()
    print("done", flush=True)
