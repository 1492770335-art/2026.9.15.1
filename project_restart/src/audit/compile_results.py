# -*- coding: utf-8 -*-
"""Compile all EXP results_raw.json + PH0/PH1 analysis into a summary markdown + noise floor.
Run after the GPU queue finishes: python src/audit/compile_results.py
"""
import json
import os

import numpy as np

ROOT = "project_restart"
EXPS = {}
for eid in os.listdir(os.path.join(ROOT, "experiments")):
    p = os.path.join(ROOT, "experiments", eid, "results_raw.json")
    if os.path.exists(p):
        try:
            EXPS[eid] = json.load(open(p, encoding="utf-8"))
        except Exception as ex:  # noqa: BLE001
            print(f"skip {eid}: {ex}")

lines = ["# 实验结果汇总（自动生成）", ""]

# per-model official scores
rows = []
for eid, d in sorted(EXPS.items()):
    if eid == "EXP-0001":
        for task, r in d.items():
            if task.startswith(("foldA_lam100", "foldB_lam100")):
                rows.append((f"Ridge {task.split('_')[0]}", r["official_score"], r["corr0"], r["corr1"], r["corr2"]))
    else:
        for task, r in d.items():
            if isinstance(r, dict) and "pooled" in r:
                rows.append((f"{eid} {task}", r["pooled"]["score"], r["pooled"]["corr0"], r["pooled"]["corr1"], r["pooled"]["corr2"]))
lines.append("| 实验/折 | Score | c0 | c1 | c2 |")
lines.append("|---|---|---|---|---|")
for name, s, c0, c1, c2 in sorted(rows):
    lines.append(f"| {name} | {s} | {c0} | {c1} | {c2} |")

# noise floor from daily scores (fold-level day heterogeneity)
lines.append("")
lines.append("## 逐日分数波动（跨日异质性，供阈值参考）")
for eid, d in sorted(EXPS.items()):
    for task, r in d.items():
        if isinstance(r, dict) and "daily" in r and r["daily"]:
            vals = [v["score"] for v in r["daily"].values()]
            lines.append(f"- {eid} {task}: daily = {vals} std={np.std(vals):.4f}")

out = os.path.join(ROOT, "reports", "RESULTS_SUMMARY.md")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))
print("written:", out)
