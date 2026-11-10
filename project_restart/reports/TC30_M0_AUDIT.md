# TC-30 Gate M0 市场状态合法性审计（只读）

- 时间：2026-11-09（Asia/Shanghai）；执行模型 deepseek-v4-pro。

## 1. 同 tick 50 股可见性（CONFIRMED，源码行号）

`script/evaluate.py`：
- L41：`frame = data.filter(...).sort(['time','code'])`；L44-45：特征按数值排序后 `reshape(-1, 50, 100)`；
- L51-53：`for tick, current in enumerate(x): prediction = model.predict(current.clone())`——`current` 形状为 **(50,100)**，即**一次调用传入同 tick 全部 50 只股票的当前特征**。

**结论**：官方接口在同一次 predict 调用中提供完整当前 tick 横截面 → 同 tick 市场状态（跨股票均值/标准差）在预测时**合法可见**，无需推断或使用未来行。M0 前提成立。

## 2. 预注册市场状态配方 MKT-v1（外层结果前冻结）

- 对每个连续特征 f_j（j=0..99，特征已标准化），在当前 tick 的 50 只股票上计算：
  1. `m_j = mean_s(f_j[s])`（市场均值）
  2. `s_j = std_s(f_j[s], ddof=1)`（市场标准差，与 torch.std 默认无偏一致）
- 输入扩展：`[f_0..f_99, m_0..m_99, s_0..s_99]` → 100 → **300 维**。
- 仅使用当前 tick 特征（无未来、无标签、无跨 tick 汇总）；每 tick 独立计算（无历史状态）；日初 reset 仅清 GRU 历史，市场统计无跨日状态。
- 统计量不做额外规范化（底层特征已标准化，统计量同尺度）。此配方冻结，不据外层结果增删。

## 3. 离线/在线一致性（CONFIRMED，单测 test_market_consistency.py）

- 训练端：`X(N,100) → reshape(T,50,100) → 每 tick 的 mean/std(ddof=1) → 广播拼接 → (N,300)`。
- 提交端：`predict(x)` 内 `x.mean(0)`/`x.std(0)`（torch 无偏）拼接 → 同一 tick 数值一致。
- 单测用合成已知 tick 验证 numpy 离线 add_market 与 torch 在线公式逐元素一致；并验证左零填充、日初 reset、无跨 tick 串扰（复用 test_window_alignment 的因果语义，K=16）。

## 4. 结论

- **M0 PASS**：接口合法、无未来/标签泄漏、离线在线一致、配方预注册。允许进入 M1（K16-GRU + MKT-v1 单因素训练）。
