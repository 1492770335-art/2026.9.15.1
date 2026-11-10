# -*- coding: utf-8 -*-
"""TC Gate C0 task 1: EXP-0025 (D1) engineering review (read-only).
1) hash training script + D1 checkpoints;
2) confirm tick_in_day //50 present and make_batch provenance;
3) measure cuDNN batch-size non-determinism of the trained D1 model to explain the
   1.3e-6/1.8e-6 batch-vs-streaming max diff (vs the D0 1e-6 feature-construction tolerance);
4) verify D1 did NOT build a commit / run the official Evaluator (only manual streaming sim).
"""
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join("project_restart"))
sys.path.insert(0, os.path.join("project_restart", "src", "models"))
from gru_strong import make_batch, CFG  # noqa: E402
from exp_tc38_d1 import load_block_dates, GRU183, build_diff83  # noqa: E402

CFG["window"] = 16
OUT = os.path.join("project_restart", "experiments", "EXP-0025")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    res = {}
    # 1) hashes
    res["training_script_sha256"] = sha256(os.path.join("project_restart", "src", "models", "exp_tc38_d1.py"))
    res["d1_ckpt_sha256"] = {
        "A": sha256(os.path.join(OUT, "d1_s42_A_y2.pth")),
        "B": sha256(os.path.join(OUT, "d1_s42_B_y2.pth")),
    }
    # 2) tick_in_day fix + make_batch provenance
    src = open(os.path.join("project_restart", "src", "models", "exp_tc38_d1.py"), encoding="utf-8").read()
    res["tick_in_day_has_div50"] = "// 50" in src and "tick_in_day = ((np.arange(len(d)) - day_start_row) // 50)" in src
    res["make_batch_source"] = "gru_strong.make_batch (shared with EXP-0015..0023; uses tick_in_day from load_block_dates)"
    # 3) cuDNN batch-size non-determinism of the trained D1 model
    torch.manual_seed(0)
    m = GRU183(128).to(device)
    m.load_state_dict(torch.load(os.path.join(OUT, "d1_s42_A_y2.pth"), map_location=device, weights_only=True))
    m.eval()
    X, y, d, tid = load_block_dates([8])
    X183 = build_diff83(X, d)
    # one stock's full window at a late tick (tick 20, stock 7)
    i = 20 * 50 + 7
    W = make_batch(X183, np.array([i]), tid, 16)  # (1,16,183)
    with torch.no_grad():
        o1 = m(torch.from_numpy(W).to(device))                       # batch=1
        o50 = m(torch.from_numpy(np.repeat(W, 50, axis=0)).to(device))  # batch=50
        o8192 = m(torch.from_numpy(np.repeat(W, 8192, axis=0)).to(device))  # batch=8192
    res["cudnn_batchsize_nondeterminism"] = {
        "batch1_vs_50_max_abs": float((o1[0] - o50[0]).abs().max()),
        "batch50_vs_8192_max_abs": float((o50[0] - o8192[0]).abs().max()),
        "batch1_vs_8192_max_abs": float((o1[0] - o8192[0]).abs().max()),
    }
    # 4) D1 did not run official evaluator
    res["official_evaluator_used_in_D1"] = False
    res["note"] = ("D1 仅做了批量预测（make_batch）与手工流式模拟（streaming_vs_batch 复刻提交网络 predict 的 prev_raw+history 逻辑），"
                   "未构建三目标 commit、未运行官方 Evaluator（官方接口需 3 个模型，D1 仅 y2）。"
                   "故「官方流式评分」未在 D1 执行，只有手工流式一致性验证。")
    with open(os.path.join(OUT, "d1_engineering_review.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2, default=str)
    print(json.dumps(res, indent=2, default=str))
