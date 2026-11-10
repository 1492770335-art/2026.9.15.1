# TC-27 新协议 K100 基线报告（TC27_K100_NEWPROTOCOL.md）

- 完成时间：2026-11-08 ~17:30（Asia/Shanghai）
- 执行模型：训练启动时为 deepseek-v4-pro；监控/收尾为 deepseek-flash（会话中途路由切换，已分别记录）
- 数据 SHA256：e03503ec…；冻结清单 reports/FREEZE_TC27_MANIFEST.json（EXP-0005/0006/0006C 共 184 文件 SHA）

## 1. 协议实施（CONFIRMED）

- 内层切分：A inner_train 0-5 / inner_stop 6-7 / full 0-7 / outer 8-11；B inner_train 0-9 / inner_stop 10-11 / full 0-11 / outer 12-15。
- 流程：inner_train 上训练，每 2 epoch 用 inner_stop 自身 pooled Pearson 选 best E（patience=4，cap=30）→ full_train 重训恰好 E epoch（seed42，shuffle rng=seed+epoch 与内层一致）→ 外层 val 仅一次评估。
- 结构/超参与 EXP-0005 完全一致（GRU128/window100/lr1e-4/wd1e-4/batch1024/clip1.0/seed42）；外层 val 未参与任何选择。

## 2. 结果（官方 pooled，CONFIRMED）

| 折 | 每目标 E (y0/y1/y2) | 优化步数 | c0 | c1 | c2 | Score |
|---|---|---|---|---|---|---|
| A（8-11） | 6 / 2 / 2 | 10974/3658/3658 | 0.4518 | 0.2850 | 0.0938 | **0.2228** |
| B（12-15） | 4 / 2 / 2 | 10976/5488/5488 | 0.4468 | 0.2847 | 0.1564 | **0.2530** |

- 逐日：A 8:0.2107 / 9:0.2117 / 10:0.2309 / 11:0.2372；B 12:0.2527 / 13:0.2571 / 14:0.2498 / 15:0.2664。
- 流式 vs 批量交叉核验：A date8 = 0.2107 == 0.2107；B date12 = 0.2527 == 0.2527（逐分一致）。

## 3. 新旧协议对比（CONFIRMED）

| 折 | 旧协议 | 新协议 | Δ |
|---|---|---|---|
| A | 0.2228（E 6/2/2） | 0.2228（E 6/2/2） | 0.0000 |
| B | 0.2531（E 6/2/2） | 0.2530（E 4/2/2） | −0.0001 |

- A 折 E 完全一致 → 分数逐分一致；B 折 y0 由旧 6 改为内层最优 4（内层 stop 峰值 0.4632@ep4，ep6 已回落），c0 0.4473→0.4468（−0.0005，四位舍入级）。
- **结论**：外层验证标签未给旧模型带来实质性选择收益（最大差 0.0001）；新协议以零代价消除早停方法学隐患，确立干净 K100 基线。

## 4. 公平性声明（重要）

- 本报告的新旧协议差**不能**用于评估窗口长度增益；K16/K64 必须在同一新协议下训练后才能与 K100 公平比较。
- A/B 仍是历史反复探索的开发日期（16–19 亦已暴露），本结果仍是开发验证口径，不得外推隐藏集。

## 5. 资源与产物

- 耗时：A 108 min / B 158 min；峰值显存 4137 MB。
- 产物：EXP-0015/{results,integrity,decision}.json、ckpt_np_{A,B}/、commit_k100_np_{A,B}/、results_raw.json。
- 失败记录：首次运行因 predict 后未恢复 train 模式报 `cudnn RNN backward can only be called in training mode`，已修复重跑，最终结果有效。
