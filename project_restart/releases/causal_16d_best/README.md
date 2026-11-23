# 图灵杯 50 股预测模型 · 训练源码

使用 20 个交易日中的前 16 天训练，后 4 天验证。每一帧包含 50 只股票的 100 个字段，每个时点为每只股票输出三个预测。冻结模型在后四天共 936,250 行上的相关系数分别为 **0.50055269、0.38622914、0.29285471**，按 20%、30%、50% 加权得到 **0.36240663**。这是多阶段组合模型在已反复用于比较的验证集上的结果。

## 准备数据与环境

在仓库根目录执行。建议使用 Linux、Python 3.9、支持 CUDA 的 PyTorch 2.5.1。原始比赛 parquet 位于仓库的 Git LFS 中。

```bash
git lfs pull --include="project/project/data/public.parquet"
cd project_restart/releases/causal_16d_best/src
python -m pip install numpy==2.0.1 scipy==1.13.1 scikit-learn==1.6.1 polars==1.36.1 lightgbm==4.6.0 pyarrow tqdm pandas
python -m joint_transformer.prepare_data --data ../../../../project/project/data/public.parquet --cache artifacts/joint_transformer/cache
```

缓存按交易日、时间、股票代码排列，形状为 `(时点数, 50, 100)`。日期从 0 开始，训练使用 0–15，验证使用 16–19。数据列 `y0、y1、y2` 顺序对应 Y1、Y2、Y3。

## 从原始数据训练

以下两个命令可以独立运行。第一个训练使用全部 50 只股票的 BookNet，第二个训练组合模型最后一层的周期回归。训练结果保存在本机的 `src/artifacts/project_restart_generalization/`。

```bash
python -u train_release.py backbone
python -u train_release.py clock
```

BookNet 同时学习个股收益、市场共同走势和未来盘口状态变化。周期回归以 20 步为一个周期，从前 8 天估计两个交易时段的相位差，再用前 16 天分别拟合市场与个股的响应。Y3 按两个标签的收益尺度承接 Y2 的修正。两步各自产生一个组件，表中指标对应完整组合。[BookNet 训练](src/experiment_phase_future_book_v1.py) · [周期层训练](src/experiment_periodic_book_response_v1.py) · [Y3 修正](src/experiment_periodic_near_carry_v1.py)

## 使用训练结果

例如，在上述 `src` 目录中，用本地训练出的 BookNet 对第 17 天产生预测并计算三个相关系数：

```python
import numpy as np
from experiment_auxiliary_quote_decoder_v1 import net
from experiment_causal_cross_horizon_ridge_v1 import CACHE, corr, labels
from experiment_phase_future_book_v1 import infer

day = 16
raw = np.load(CACHE / f'day_{day:02d}_x.npy', mmap_mode='r')
pred = infer(net(16), raw)
print(pred.shape)
print(corr(pred.reshape(-1, 3), labels(day)))
```

完整模型还依赖早期 GRU、树模型、滚动窗口预测、专家融合和事件先验。训练早期组件后，需在训练期内依次生成历史窗口的样本外预测，供后续校准使用。各研究脚本中的 `OUT`、`load`、`train` 和 `screen` 定义对应的输入、输出与筛选过程。项目所用 FI-2010 来源和预处理定义在外部先验训练源码中。完整组合依赖这些阶段各自生成的检查点与中间产物；仅运行以上两条命令可复现两个组件的训练。[基础专家](src/experiment_peer_dual_temporal.py) · [树模型](src/experiment_pairwise_book_tree_v1.py) · [会话专家](src/replay_session_experts_16d.py) · [外部数据处理](src/experiment_fi2010_transfer_v1.py) · [事件先验](src/experiment_external_event_prior_v1.py) · [组合顺序](src/released_model.py)

## 周期校正强度的训练和验证

第一阶段在第 9–16 天的历史滚动预测上分别估计三个目标的校正强度。在同一切分上，综合分由 0.36227023 提升至 0.36235740。Y1 和 Y2 使用各自的系数，Y3 使用前一版系数。

在 `src` 目录运行以下命令。需先生成 `train_release.py clock` 的输出、第 9–16 天的两组样本外基线预测，以及用于完整回放的上游模型。脚本先拟合系数，再对第 17–20 天的 936,250 行做完整回放和截断输入检查。权重和预测缓存由本机生成。[强度估计与训练](src/experiment_periodic_response_strength_v1.py) · [回放入口](src/replay_periodic_parent_v1.py)

```bash
python -u experiment_periodic_response_strength_v1.py
```

## 个股预测修正

第二阶段用当前周期位置、个股三个期限的预测以及最近 5 步和 20 步的预测变化，修正 Y1、Y2。先完成上一节的训练，再运行下列命令。综合分从 0.36235740 提升至 0.36240663，四天单日分数依次为 0.409428、0.307301、0.350832、0.372108。重载参数和重新组合上游模型得到了相同的三个目标相关系数。[训练与验证入口](src/experiment_phase_conditioned_forecasts_v1.py)

```bash
python -u experiment_phase_conditioned_forecasts_v1.py
```

每个经同一切分全量复核后创下新高的版本在发布分支上对应一个独立提交。提交保留训练代码、使用说明和验证分数。
