"""Matched y2 GRU128 vs GRU64 comparison on the two earlier time folds.

One preselected change: hidden dimension 128 -> 64. Both runs use the frozen
K16/seed42/two-epoch EXP-0016 recipe; dates 16-19 are not used for selection.
"""

import gc
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.nn import functional as F


ROOT = Path(__file__).resolve().parent
REPO = ROOT / "2026.9.15.1"
DATA = ROOT / "project/data/public.parquet"
OUT = ROOT / "artifacts/project_restart_generalization/hidden64_pilot"
EXPECTED_SHA = "e03503ec4d519c043a5b95c4dd6684f25d2636dd00a1d015ba3128b1d7818556"
FOLDS = {
    "A": {"train": list(range(8)), "validation": [8, 9, 10, 11]},
    "B": {"train": list(range(12)), "validation": [12, 13, 14, 15]},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for part in iter(lambda: f.read(1 << 20), b""):
            h.update(part)
    return h.hexdigest()


def save_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    tmp.replace(path)


def main():
    if sha256(DATA) != EXPECTED_SHA:
        raise RuntimeError("Data SHA-256 mismatch")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    OUT.mkdir(parents=True, exist_ok=True)
    os.chdir(REPO)
    sys.path[:0] = [str(REPO / "project_restart/src/models"),
                   str(REPO / "project_restart"), str(REPO / "project/project")]
    import gru_strong as src
    from src.eval.metrics import pooled_pearson_raw

    src.DATA = str(DATA)
    src.CFG["window"] = 16
    torch.set_num_threads(4)
    device = "cuda:0"
    plan = {
        "source_commit": "9af3462a5d49fcba8d788c666d68e7a29e9ddf63",
        "data_sha256": EXPECTED_SHA,
        "source_sha256": sha256(REPO / "project_restart/src/models/gru_strong.py"),
        "hypothesis": "Smaller y2 GRU hidden dimension improves held-out dates by reducing capacity",
        "folds": FOLDS,
        "target": 2, "window": 16, "seed": 42, "epochs": 2,
        "batch_size": 1024, "lr": 1e-4, "weight_decay": 1e-4,
        "hidden_values": [128, 64],
        "promotion_gate": "raw c2 gain >= 0.008 on both folds and >= 6/8 daily gains nonnegative",
        "outer_dates": [16, 17, 18, 19],
        "outer_usage": "not used in this experiment",
    }
    plan_path = OUT / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Output directory belongs to another plan")
    save_json(plan_path, plan)
    result = {"plan": str(plan_path), "folds": {}}

    for fold, spec in FOLDS.items():
        print(json.dumps({"stage": "load", "fold": fold, "train": spec["train"],
                          "validation": spec["validation"]}), flush=True)
        Xtr, ytr, ttr = src.load_block(spec["train"])
        Xva, yva, tva = src.load_block(spec["validation"])
        dval = (pl.scan_parquet(DATA).filter(pl.col("date").is_in(spec["validation"]))
                .sort(["date", "time", "code"]).select("date").collect()["date"].to_numpy())
        if len(dval) != len(Xva):
            raise RuntimeError("Date alignment mismatch")
        print(json.dumps({"stage": "loaded", "fold": fold, "train_rows": len(Xtr),
                          "val_rows": len(Xva)}), flush=True)
        result["folds"][fold] = {}
        for hidden in (128, 64):
            torch.manual_seed(42)
            model = src.GRU(hidden=hidden).to(device)
            Y = torch.from_numpy(ytr[:, 2:3]).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=src.CFG["lr"],
                                     weight_decay=src.CFG["weight_decay"])
            losses = []
            started = time.perf_counter()
            n_batches = math.ceil(len(Xtr) / 1024)
            for epoch in range(2):
                model.train()
                order = np.random.default_rng(42 + epoch).permutation(len(Xtr))
                total = 0.0
                for batch, start in enumerate(range(0, len(order), 1024), 1):
                    indices = order[start:start + 1024]
                    x = torch.from_numpy(src.make_batch(Xtr, indices, ttr, 16)).to(device)
                    loss = F.mse_loss(model(x), Y[indices])
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    total += loss.detach().item() * len(indices)
                    if batch == 1 or batch % 500 == 0 or batch == n_batches:
                        print(json.dumps({"stage": "train", "fold": fold,
                                          "hidden": hidden, "epoch": epoch + 1,
                                          "batch": batch, "batches": n_batches,
                                          "mean_loss": total / (min(batch * 1024, len(Xtr)))}),
                              flush=True)
                losses.append(total / len(Xtr))
            pred = src.predict_batched(model, Xva, tva, device)
            corr = float(pooled_pearson_raw(pred, yva[:, 2:3])[0])
            daily = {}
            for day in spec["validation"]:
                mask = dval == day
                daily[str(day)] = float(pooled_pearson_raw(pred[mask], yva[mask, 2:3])[0])
            ckpt = OUT / f"fold{fold}_hidden{hidden}_s42_y2.pth"
            torch.save(model.state_dict(), ckpt)
            record = {"hidden": hidden, "corr2": corr, "daily_corr2": daily,
                      "train_losses": losses, "parameters": sum(p.numel() for p in model.parameters()),
                      "elapsed_seconds": round(time.perf_counter() - started, 1),
                      "checkpoint": str(ckpt), "checkpoint_sha256": sha256(ckpt)}
            result["folds"][fold][str(hidden)] = record
            save_json(OUT / "results_partial.json", result)
            print(json.dumps({"stage": "model_complete", "fold": fold,
                              "hidden": hidden, "corr2": corr, "daily": daily,
                              "seconds": record["elapsed_seconds"]}), flush=True)
            del model, Y, opt, pred
            gc.collect()
            torch.cuda.empty_cache()
        del Xtr, ytr, ttr, Xva, yva, tva, dval
        gc.collect()
        torch.cuda.empty_cache()
    for fold in FOLDS:
        pair = result["folds"][fold]
        pair["delta_corr2"] = pair["64"]["corr2"] - pair["128"]["corr2"]
        pair["daily_delta_corr2"] = {str(day): pair["64"]["daily_corr2"][str(day)]
                                      - pair["128"]["daily_corr2"][str(day)]
                                      for day in FOLDS[fold]["validation"]}
    daily_nonnegative = sum(delta >= 0 for fold in FOLDS
                            for delta in result["folds"][fold]["daily_delta_corr2"].values())
    result["promotion_gate_passed"] = (all(result["folds"][fold]["delta_corr2"] >= 0.008
                                           for fold in FOLDS) and daily_nonnegative >= 6)
    result["daily_nonnegative_count"] = daily_nonnegative
    save_json(OUT / "results.json", result)
    print(json.dumps({"stage": "complete", "delta_A": result["folds"]["A"]["delta_corr2"],
                      "delta_B": result["folds"]["B"]["delta_corr2"],
                      "daily_nonnegative": daily_nonnegative,
                      "passed": result["promotion_gate_passed"]}), flush=True)


if __name__ == "__main__":
    main()
