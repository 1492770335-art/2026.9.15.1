# 图灵杯高频预测提交说明（三种子目标路由）

## 模型结构
三个目标使用不同网络（README 允许三目标不同架构）：
- y0：MarketGRU K16（GRU128，隐藏 128；输入 300 = 100 个股特征 + 当前 tick 横截面 mean/std），seed42/43/44 三个模型预测 1:1:1 等权平均。
- y1：同 y0（MarketGRU K16，三 seed 等权）。
- y2：非 Market K16-GRU（GRU128，隐藏 128；输入 100 原始特征），seed42/43/44 三个模型预测 1:1:1 等权平均。

## 训练
- 数据：公开 date0–19（全 20 日）；每个目标独立训练；y0 固定 4 epoch、y1 固定 2 epoch、y2 固定 2 epoch（epoch 数由 0–15 内部验证预注册，不在 full20 重搜）。
- 优化：AdamW lr=1e-4、weight_decay=1e-4；MSE 损失；batch=1024；梯度裁剪 1.0；seed=42/43/44。
- 因果性：每股票同日因果窗口 K=16，左零填充，日初 reset；市场统计仅用当前 tick 50 股特征（mean/std，ddof=1），无未来/标签信息。

## 加载
- Model_0/1 的 .pth 各含 members.0/1/2 三个 seed 的 MarketGRU 权重；Model_2 含三个 seed 的 K16-GRU 权重。
- 推理仅需 torch（网络类仅用 torch.nn/torch.nn.functional）；输入按官方接口 float32 [50,100]，输出 [50,1]。

## 依赖
torch（参见 requirements.txt：numpy/polars/torch/tqdm）。
