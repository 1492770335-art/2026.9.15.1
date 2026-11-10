# diff83 因果差分特征规格（diff83_spec.md）

## 目标
为 K16-GRU y2 追加相邻 tick 差分特征（方法假设 H-DIFF83）：对连续特征 f0–f82 计算同一股票、同一交易日的相邻 tick 变化，与原始 100 维拼接成 **183 维**输入。

## 定义
- `delta_f(t, stock) = f(t, stock) - f(t-1, stock)`，仅对 f0–f82（83 个连续特征，列 0..82）。
- 交易日第一个真实 tick 的差分 = 0（无前一 tick）。
- 不对 f83–f99 离散类别做数值差分。
- 严格按 (date, time, code) 行序、每股票 stride=50；禁止跨股票、跨日期串联。
- 只用当前与已到达历史 tick，无未来、无标签。

## 关键历史长度（17-tick 边界）
K16 窗口含 16 个原始 tick；要正确计算窗口**第一个位置**的差分，需要再往前 1 个 tick（共 17 个原始 tick）。实现上：
- 离线：全序列逐 tick 计算 diff（同日同股），窗口切片时最旧 tick 的 diff 已由前一个 tick 正确算出，**不在滑窗第一格重新置零**。
- 在线：网络维护 `prev_raw`（上一 tick 原始 100 维，日初 None → 首 tick diff=0），与 16-tick 的 183 维历史缓存分离；窗口最旧 tick 的 diff 即其被处理时刻用 `prev_raw` 算得的值。

## 离线 / 在线一致
- 离线 `build_diff83(X_raw, dates)`：`diff[same] = X_raw[same,:83] - X_raw[prev,:83]`（same = 同日且 idx>=50），否则 0；返回 `concat(raw, diff)` (N,183)。
- 在线 `online_feat183(x_t, prev_raw)`：`delta = x_t[:,:83]-prev_raw[:,:83]`（prev_raw=None 时 0）；返回 concat。
- 单测（test_diff83.py）验证逐 tick 对齐，float32 max abs diff = 0（容差 ≤1e-6）。

## 参数量与资源（GRU128，输入 100→183）
- GRU 参数：weight_ih (3·h·input) + weight_hh (3·h·h) + bias_ih (3·h) + bias_hh (3·h)。
  - 100 维：3·128·100=38400 + 49152 + 768 = 88320；+ output Linear 129 → ≈88449。
  - 183 维：3·128·183=70272 + 49152 + 768 = 120192；+ output 129 → ≈120321。
  - **增量 = +31872 参数/GRU（仅输入→隐层权重增长，隐层→隐层与输出不变）**。
- 数据内存：每样本 183/100 = 1.83×。Fold A（8 日）约 1.37GB、Fold B（12 日）约 2.06GB、full20 约 3.42GB（float32）。显著小于 Market 300 维的 OOM 风险，但沿用 S1 分阶段加载。
- GPU 显存：训练 batch 1024×16×183×4 ≈ 12MB；流式逐 tick 50×183，GRU183 前向约为 100 维的 ~1.4×（单模型 ~7s→~10s/日量级）。
- 说明：差分是 K16 可见序列的**确定性表示**，非新增独立信息；若 D1 有收益，可能来自优化或输入维数带来的容量变化，不能直接归因为"发现新信息"（D2 可加等容量 placebo 对照）。
