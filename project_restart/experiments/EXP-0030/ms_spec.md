# K16+K4 多尺度局部时间分支规格（ms_spec.md）

## 架构（M0 冻结，唯一候选）
- **主干**：原 K16-GRU128，输入 100 维不变（f0–f99，f83–f99 保持序数数值）；取最终隐状态 h（128 维）。
- **局部短尺度分支**（同一 K16 窗口内，仅最近 4 tick）：
  1. `DepthwiseConv1d(100, 100, kernel_size=4, groups=100, bias=True)`——每特征通道独立的时序卷积；
  2. `ReLU`（固定激活）；
  3. 取最后一个时间位置（汇总 t−3..t 的 4 tick）→ (100,)；
  4. `Linear(100, 32, bias=True)` → 32 维局部表示。
- **预测头**：`Linear(160, 1, bias=True)` 作用于 `concat[h(128), local(32)]` → y2。
- 初始化：PyTorch 默认（conv/linear 默认均匀、GRU 默认）。所有 bias=True。无 dropout/正则新增。

## 因果性 / 无泄漏
- 卷积 kernel=4、无 padding，取最后一个位置 → 只用 `[x(t-3), x(t-2), x(t-1), x(t)]`，**不含未来**；
- 左零填充：日初不足 4 个真实 tick 时，buffer 以 134 维…（此处 100 维）零补齐；训练（make_batch 零填充）与流式（4-tick buffer 左填充）一致；
- 日初 reset：GRU 历史与 4-tick buffer 均按日清空；跨日/跨股票不串联；50 股顺序逐行独立。

## 参数与资源
- GRU(100,128) = 88320；DepthwiseConv = 500（400 w + 100 b）；local_proj = 3232；head = 161；**总计 92213**。
- 基线 GRU100 = 88320 + 129 = 88449 → **增量 +3764 参数（约 +4.3%）**。
- 数据内存不变（仍 100 维）；卷积/投影开销极小（depthwise 100 通道、kernel4），流式每 tick 增加可忽略计算。

## 容量混淆声明
- 局部分支输入都是 K16 窗口的确定性函数，非新增独立信息；新增 +3764 参数带来容量变化，若 M1 有增益需等容量 placebo 排除。

## 单测（test_k4_branch.py，10/10 PASS）
- 参数精确、离线/流式逐元素对齐（max abs diff = 0）、因果隔离（未来 tick 扰动不改过去）、日初有限性、50 股隔离。
