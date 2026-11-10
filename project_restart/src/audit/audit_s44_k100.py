# -*- coding: utf-8 -*-
"""EXP-0018A: read-only anomaly audit of Fold A seed44 K100 new-vs-old protocol (0.2048 vs 0.2263).
No training, no weight changes. Compares checkpoint SHA, per-target official corr, and decomposes gap.
"""
import hashlib
import json
import os
import sys

import torch

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project", "project"))
from script.evaluate import Evaluator  # noqa: E402

DATA = os.path.join("project", "project", "data", "public.parquet")
OLD_COMMIT = os.path.join("project_restart", "experiments", "EXP-0006", "commit_gru_A_s44")
NEW_COMMIT = os.path.join("project_restart", "experiments", "EXP-0017", "commit_s44_K100_A")
OLD_CKPT = os.path.join("project_restart", "experiments", "EXP-0006", "ckpt_gru_A_s44")
NEW_CKPT = os.path.join("project_restart", "experiments", "EXP-0017", "ckpt_s44_K100_A")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def eval_commit(commit, dates):
    ev = Evaluator(DATA, commit, "cuda:0")
    return ev.evaluate(date=dates)


if __name__ == "__main__":
    res = {}
    # checkpoint SHA comparison per target
    res["checkpoint_sha"] = {}
    for label in range(3):
        op = os.path.join(OLD_CKPT, f"Model_{label}.pth")
        np_ = os.path.join(NEW_CKPT, f"Model_{label}.pth")
        res["checkpoint_sha"][f"y{label}"] = {
            "old_sha": sha256(op)[:16] if os.path.exists(op) else "MISSING",
            "new_sha": sha256(np_)[:16] if os.path.exists(np_) else "MISSING",
            "identical": sha256(op) == sha256(np_) if (os.path.exists(op) and os.path.exists(np_)) else None,
        }
    # evaluate both on outer val 8-11
    res["old_official"] = eval_commit(OLD_COMMIT, [8, 9, 10, 11])
    res["new_official"] = eval_commit(NEW_COMMIT, [8, 9, 10, 11])
    # daily for new
    res["new_daily"] = {d: eval_commit(NEW_COMMIT, d) for d in [8, 9, 10, 11]}
    res["old_daily"] = {d: eval_commit(OLD_COMMIT, d) for d in [8, 9, 10, 11]}
    gap = res["new_official"]["score"] - res["old_official"]["score"]
    c2_gap = res["new_official"]["corr2"] - res["old_official"]["corr2"]
    res["analysis"] = {
        "score_gap": gap, "c0_gap": res["new_official"]["corr0"] - res["old_official"]["corr0"],
        "c1_gap": res["new_official"]["corr1"] - res["old_official"]["corr1"],
        "c2_gap": c2_gap, "c2_weighted_contribution": 0.5 * c2_gap,
        "note": "gap 应主要来自 c2（y2 的 E 由旧 2 → 新 4）"
    }
    out = os.path.join("project_restart", "experiments", "EXP-0018", "audit_s44_k100.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))
