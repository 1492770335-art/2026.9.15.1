# -*- coding: utf-8 -*-
"""EXP-0016 (EXP-0007-NP): new-protocol GRU window ablation K in {16,64}, seed42, folds A/B.
K=100 reference = EXP-0015 (0.2228 / 0.2530), not retrained here.
Same structure/hyperparams as strong-GRU (hidden128/lr1e-4/wd1e-4/batch1024/clip1.0/seed42).
New inner early-stop protocol (identical to TC-27): inner_stop selects E per target; full_train
retrains fixed E; outer_val evaluated once. train/eval bugfix (model.train() at epoch start).
"""
import json
import math
import os
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
sys.path.insert(0, os.path.join("project_restart", "src", "models"))
from script.evaluate import Evaluator  # noqa: E402
from gru_strong import load_block, make_batch, GRU, predict_batched, build_commit, CFG  # noqa: E402
from src.eval.metrics import pooled_pearson_raw, official_score  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
OUT = os.path.join("project_restart", "experiments", "EXP-0016")
os.makedirs(OUT, exist_ok=True)

CAP = 30
PATIENCE = 4
EVAL_EVERY = 2
SEED = 42
K100_REF = {"A": 0.2228, "B": 0.2530}

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


def run_window(K, fold):
    CFG["window"] = K
    spec = FOLDS[fold]
    t0 = time.time()
    Xi, yi, tii = load_block(spec["inner_train"])
    Xs, ys, tis = load_block(spec["inner_stop"])
    Xf, yf, tif = load_block(spec["full_train"])
    ckpt = os.path.join(OUT, f"ckpt_K{K}_{fold}")
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
        print(f"[K{K} {fold} y{label}] E={E} steps={steps}", flush=True)
    commit_dir = os.path.join(OUT, f"commit_K{K}_{fold}")
    build_commit(sds, commit_dir)
    ev = Evaluator(DATA, commit_dir, device)
    pooled = ev.evaluate(date=spec["outer_val"])
    daily = {int(d): ev.evaluate(date=d) for d in spec["outer_val"]}
    extra = {}
    if fold == "A":
        extra["no_date8"] = ev.evaluate(date=[9, 10, 11])
    # batched-vs-streaming + latency on first outer val date
    Xv, yv, tiv = load_block([spec["outer_val"][0]])
    import importlib.util
    sm = importlib.util.spec_from_file_location("netw3", os.path.join(commit_dir, "network.py"),
                                                submodule_search_locations=[os.path.abspath(commit_dir)])
    netw3 = importlib.util.module_from_spec(sm)
    sys.modules["netw3"] = netw3
    sm.loader.exec_module(netw3)
    models = []
    for label in range(3):
        cfg = json.load(open(os.path.join(commit_dir, "conf", f"Model_{label}.json"), encoding="utf-8"))
        m = getattr(netw3, cfg["model"])(cfg).to(device)
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
                Xg = torch.from_numpy(make_batch(Xv, I, tiv, K)).to(device)
                o.append(m(Xg).cpu().numpy())
            P.append(np.concatenate(o))
    P = np.stack(P, axis=1).reshape(-1, 3)
    batched = official_score(P, yv)
    tlat0 = time.time()
    _ = ev.evaluate(date=spec["outer_val"][0])
    latency = time.time() - tlat0
    rec = {"K": K, "fold": fold, "per_target": info, "pooled_official": pooled,
           "daily": daily, "extra": extra, "batched_check_first_val_day": batched,
           "streaming_latency_sec_day": round(latency, 1),
           "delta_vs_K100": round(pooled["score"] - K100_REF[fold], 4),
           "elapsed_sec": round(time.time() - t0, 1),
           "peak_vram_mb": round(torch.cuda.max_memory_allocated(device) / 1e6, 1)}
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    return rec


if __name__ == "__main__":
    torch.set_num_threads(4)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print("device:", device, "base CFG:", {k: CFG[k] for k in ("hidden", "lr", "epochs", "batch_size", "seed")}, flush=True)
    results = {}
    for K in (16, 64):
        for fold in ("A", "B"):
            rec = run_window(K, fold)
            results[f"K{K}_{fold}"] = rec
            with open(os.path.join(OUT, "results_raw.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)
            print(json.dumps({k: rec[k] for k in ("K", "fold", "per_target", "pooled_official", "daily", "extra", "delta_vs_K100", "streaming_latency_sec_day", "elapsed_sec", "peak_vram_mb")}, indent=2, default=str), flush=True)
    print("done", flush=True)
