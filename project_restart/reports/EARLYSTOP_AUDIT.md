# 早停协议审计与修订提案（EARLYSTOP_AUDIT.md）

- 时间：2026-11-08（Asia/Shanghai）
- 执行模型：deepseek-flash（Harness 实际路由；要求 v4-pro/max 未果，如实记录）
- 范围：只读源码核验 + 协议设计；未重新训练任何模型。

## 1. 审计结论（CONFIRMED，源码行号证据）

**EXP-0005 / EXP-0006（strong-GRU）确实使用外层 A/B 验证标签选择最优 epoch。**

证据链：
1. `src/models/gru_strong.py` L106-113（train_one）：
   - L107：`p = predict_batched(model, Xva, tick_in_day_va, device)` —— Xva/yva 即外层验证折（由 run_task 传入：Fold A = dates 8-11，Fold B = dates 12-15）；
   - L108：`c = pooled_pearson_raw(p, yva[:, label:label+1])[0]` —— 用外层验证标签计算自身目标 pooled Pearson；
   - L110-111：以该分数选择 best epoch 并保存 checkpoint；
   - L112-113：patience 同样由该分数驱动。
2. EXP-0006b（exp0006b_seeds.py）复用同一 train_one → 同样口径。
3. `src/models/mlp_strong.py` train_one 同构（L106-113 对应段，eval 于 Xva 选 epoch）→ strong-MLP 亦然。
4. 不受影响的实验：EXP-0003（官方配方固定 2 epoch，无选择）；EXP-0001 Ridge（λ=100 为预注册默认，λ 网格仅报告未选择）。

**后果**：date 8-11 / 12-15 已参与 checkpoint 选择 → 现有 A/B 分数应标注为「开发验证 CV」（对 epoch 选择有乐观偏差），不能视为对外层验证的干净估计，更不能外推为隐藏集无偏估计。历史分数保留不变，仅标注口径。

## 2. 内部早停协议提案（仅设计，待授权后实施）

目标：外层 val 仅用于最终一次性评估；epoch 选择完全在训练日期内部完成；K100/K16/K64 同口径公平比较。

### 2.1 内部切分（时间前推）

| fold | 内部训练 train_i | 内部停训 stop_i | 最终全量训练 | 外层评估（仅一次） |
|---|---|---|---|---|
| A | date 0–5 | date 6–7 | date 0–7 | date 8–11 |
| B | date 0–9 | date 10–11 | date 0–11 | date 12–15 |

### 2.2 流程（逐目标 y0/y1/y2 独立）

1. 在 train_i 上训练，每 `eval_every=2` epoch 用 stop_i 的自身目标 pooled Pearson（批量推理，与流式语义一致）选 best epoch E（patience=4，epoch 上限 30）。
2. 用同一 seed、同一超参在**全量训练日期**上重训恰好 E 个 epoch（shuffle 种子沿用 seed+epoch 规则），不再做任何选择。
3. 外层 val 仅一次评估：官方 Evaluator pooled + 逐日。
4. 多 seed：每 seed 独立走 1–3；三种子 1:1:1 预测平均作为 ensemble 组合（EXP-0006C 已证明该组合有效且无权重学习）。

### 2.3 K 公平性规则

- K ∈ {16, 64, 100} 必须全部在本协议下重跑（含 K=100 新对照）；EXP-0005/0006 的 K100 checkpoint 因外层选择口径不同**不能充当公平对照**。
- 仅变量 = 窗口 K；hidden/lr/epoch 上限/batch/seed/内部切分/评估口径完全一致。
- 双头 GRU（P1-A）的 μ=0 基线也必须在本协议下复现后才能对比 μ>0。

### 2.4 记录要求

每个候选记录：内部 best epoch E、内部 stop_i 分数轨迹、外层 pooled c0/c1/c2、raw/官方 Score、逐日、耗时/显存、预测与权重 SHA。外层分数仍为开发验证（仅 4 天、反复使用），16–19 仍仅冻结后附加检查。

## 3. 与现行结果的衔接

- 现行 EXP-0005/0006 分数标注为「旧协议开发验证」，用于探索性比较；
- EXP-0006C 融合诊断在旧协议 checkpoint 上完成（零训练），其结论（融合增益）也需在新协议复训后重新验证；
- 任何晋级（KEEP 候选）须以新协议分数为准。
