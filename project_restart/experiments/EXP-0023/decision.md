# EXP-0023 决策记录（TC-34 Gate S2 最终交付）

- **一句话结论**：冻结 9 子模型路由在 date0–19 全量重训（y0=4/y1=2/y2=2 固定 E），打包为独立提交目录 + zip（SHA 6a3fba99…），官方 Evaluator 从独立副本跑完 20 日全 tick，工程验收通过。
- **判定**：**READY_FOR_USER_SUBMISSION**（工件已冻结/打包/验收，未上传平台，不宣称隐藏成绩）。
- **工程证据**：20 日官方流式无 NaN/Inf、每日 reset、输出 (50,1)；逐日 18.6–25.3s、全程 394s；显存 allocated 213.6MB/reserved 228.6MB/nvidia-smi 403MB；20 日 in-sample Score 0.3916（IN_SAMPLE_ENGINEERING_DIAGNOSTIC）。
- **失败/注意**：无运行失败（S1 的 OOM 修复已在 S2 分阶段加载中沿用）；RAM 峰值未测量（psutil 不可用，已注明）。
- **回退候选**：EXP-0018 非 Market K16 三 seed 全目标融合（A 0.2495/B 0.2691）保留，未覆盖。
- **交付物**：releases/turing_submission_v1/（base.py、network.py、conf×3、model×3、description.md、manifest.json）与 turing_submission_v1.zip。
