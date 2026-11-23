"""Reproduce the repository's frozen K16 routing on date 0-15, check 16-19.

Uses the original model, optimizer, batching, and official evaluator. Market
features are gathered lazily to keep the 300-column array out of RAM.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
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
DEFAULT_OUT = ROOT / "artifacts/project_restart_k16_reproduction"
EXPECTED_DATA_SHA = "e03503ec4d519c043a5b95c4dd6684f25d2636dd00a1d015ba3128b1d7818556"


class LazyMarketArray:
    """Same float32 [own, tick mean, tick sample std] as original add_market."""

    def __init__(self, raw):
        if raw.ndim != 2 or raw.shape[1] != 100 or len(raw) % 50:
            raise ValueError("Expected complete sorted 50-stock ticks")
        self.raw = raw
        grid = raw.reshape(-1, 50, 100)
        self.mean = grid.mean(axis=1)
        self.std = grid.std(axis=1, ddof=1)
        self.shape = (len(raw), 300)

    def __len__(self):
        return len(self.raw)

    def __getitem__(self, index):
        return np.concatenate((self.raw[index], self.mean[index // 50],
                               self.std[index // 50]), axis=-1)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str))
    temporary.replace(path)


def prepare_source(out):
    os.chdir(REPO)
    sys.path.insert(0, str(REPO / "project_restart/src/models"))
    sys.path.insert(0, str(REPO / "project/project"))
    sys.path.insert(0, str(REPO / "project_restart"))
    import gru_strong
    import exp_tc34_s2 as source

    gru_strong.DATA = str(DATA)
    source.OUT = str(out)
    source.ALL_DATES = list(range(16))
    source.add_market = LazyMarketArray
    source.CFG["window"] = 16
    return source


def install_progress_logging(source):
    context = {"model": "", "epoch": 0}

    def train_epoch_fn(model, X, Y, tid, order, opt, device):
        model.train()
        context["epoch"] += 1
        total, cnt = 0.0, 0
        batches = math.ceil(len(order) / source.CFG["batch_size"])
        started = time.perf_counter()
        for batch, offset in enumerate(range(0, len(order), source.CFG["batch_size"]), 1):
            indices = order[offset:offset + source.CFG["batch_size"]]
            window = torch.from_numpy(source.make_batch(X, indices, tid, 16)).to(device)
            loss = F.mse_loss(model(window), Y[indices])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), source.CFG["clip"])
            opt.step()
            total += loss.detach().item() * len(indices)
            cnt += len(indices)
            if batch == 1 or batch % 250 == 0 or batch == batches:
                elapsed = time.perf_counter() - started
                print(json.dumps({"stage": "training", "model": context["model"],
                                  "epoch": context["epoch"], "batch": batch,
                                  "batches": batches, "mean_loss": total / cnt,
                                  "elapsed_seconds": round(elapsed, 1),
                                  "epoch_eta_seconds": round(elapsed / batch * (batches - batch), 1)}),
                      flush=True)
        return total / cnt

    source.train_epoch_fn = train_epoch_fn
    return context


def package_commit(source, out):
    commit = out / "commit_k16_routing"
    (commit / "conf").mkdir(parents=True, exist_ok=True)
    (commit / "model").mkdir(parents=True, exist_ok=True)
    release = REPO / "project_restart/releases/turing_submission_v1"
    shutil.copy2(release / "base.py", commit / "base.py")
    shutil.copy2(release / "network.py", commit / "network.py")
    for label, prefix in ((0, "market"), (1, "market"), (2, "k16")):
        shutil.copy2(release / "conf" / f"Model_{label}.json",
                     commit / "conf" / f"Model_{label}.json")
        members = {}
        for i, seed in enumerate((42, 43, 44)):
            path = out / f"{prefix}_full_s{seed}_y{label}.pth"
            state = torch.load(path, map_location="cpu", weights_only=True)
            for key, value in state.items():
                members[f"members.{i}.{key}"] = value
        torch.save(members, commit / "model" / f"Model_{label}.pth")
    return commit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if sha256(DATA) != EXPECTED_DATA_SHA:
        raise RuntimeError("Local parquet differs from repository experiment data")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this full-data reproduction")
    torch.set_num_threads(4)
    device = "cuda:0"
    source = prepare_source(out)
    context = install_progress_logging(source)
    plan = dict(repo_sha="a196b0b10d8ff947669b01b02c681110b0fbc198",
                data_sha256=EXPECTED_DATA_SHA, train_dates=list(range(16)),
                check_dates=list(range(16, 20)), window=16,
                epochs={"y0": 4, "y1": 2, "y2": 2}, seeds=[42, 43, 44],
                batch_size=source.CFG["batch_size"], lr=source.CFG["lr"],
                weight_decay=source.CFG["weight_decay"],
                source_sha256=sha256(REPO / "project_restart/src/models/exp_tc34_s2.py"),
                runner_sha256=sha256(__file__),
                note="Frozen final recipe on 0-15, check 16-19. No inner epoch search. "
                     "Market features use lazy equivalent computation. Prior studies "
                     "already exposed check dates; not a blind test.")
    plan_path = out / "plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError("Existing output has a different plan")
    atomic_json(plan_path, plan)
    result_path = out / "results_partial.json"
    results = json.loads(result_path.read_text()) if result_path.exists() else {"train": []}
    done = {(r["seed"], r["label"]) for r in results["train"]}
    started = time.perf_counter()
    for market in (True, False):
        print(f"[data] loading dates 0-15, market={market}", flush=True)
        X, y, tid = source.load_full(market)
        print(f"[data] rows={len(X)} features={X.shape[1]} market={market}", flush=True)
        for seed in (42, 43, 44):
            for label, epochs in (((0, 4), (1, 2)) if market else ((2, 2),)):
                if (seed, label) in done:
                    continue
                context.update(model=f"{'market' if market else 'k16'}_s{seed}_y{label}", epoch=0)
                record = source.train_full(seed, label, market, epochs, X, y, tid, device)
                results["train"].append(record)
                atomic_json(result_path, results)
                print(json.dumps({"stage": "model_complete", **record}), flush=True)
        del X, y, tid
        gc.collect()
        torch.cuda.empty_cache()
    commit = package_commit(source, out)
    from script.evaluate import Evaluator
    evaluator = Evaluator(str(DATA), str(commit), device)
    print("[evaluation] official streaming dates 16-19", flush=True)
    pooled = evaluator.evaluate(date=[16, 17, 18, 19])
    daily = {str(day): evaluator.evaluate(date=day) for day in (16, 17, 18, 19)}
    rows = pl.scan_parquet(DATA).filter(pl.col("date").is_in([16, 17, 18, 19])).select(
        pl.len()).collect().item()
    results["check"] = dict(pooled_official=pooled, daily_official=daily,
                            rows=rows)
    results["total_seconds"] = time.perf_counter() - started
    atomic_json(out / "results.json", results)
    print(json.dumps({"stage": "complete", "check": results["check"],
                      "total_seconds": results["total_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
