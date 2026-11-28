# 跨股票训练误差校正模型

用前 16 天训练，以后 4 天的 936,250 条股票观测验证。将四天的预测合并，对每个目标计算 Pearson 相关系数，再以 0.2、0.3、0.5 加权。上一正式版为 `5006dc8ee454ed84304a92c034a51c5575488fd4`。

| 指标 | 上一正式版 | 本版 |
| --- | ---: | ---: |
| Y1 相关系数 | 0.50217681 | 0.50217681 |
| Y2 相关系数 | 0.39202457 | 0.39310596 |
| Y3 相关系数 | 0.31017993 | 0.31164268 |
| 第 17 天综合分 | 0.41131684 | 0.41181420 |
| 第 18 天综合分 | 0.33672366 | 0.33833433 |
| 第 19 天综合分 | 0.36307365 | 0.36576332 |
| 第 20 天综合分 | 0.37070923 | 0.37082826 |
| 四天合并综合分 | 0.3731327005317729 | **0.3741884928047098** |

累计提高 0.001055792272936934。Y3 贡献约 0.00073137，Y2 贡献约 0.00032442。第 19 天的综合分改善最多。以上日期已多次用于版本比较，真正的新日期仍需另行检验。

## 模型如何使用跨股票信息

上游沿用上一版的 50 股三目标预测。本次积累的 19 层依次对 Y2 或 Y3 做校正，包括市场共同波动、股票间定价关系、价格重访、跨股票事件时间及已到期预测的表现。每层先在前 16 天内做按日期推进的筛选，保留通过筛选的目标分支。完整顺序及各层训练期选中的分支见 [layers.py](layers.py)。

最后一层针对 Y3。将 50 只股票分为两个互不重叠的组，用一组内股票的多空组合跟踪另一组的共同变动。训练时把已经实现的组合误差和本组共同走势误差作为控制变量，在不同训练日期上估计和应用校正系数。两组训练日期给出同向的预测信号才保留。推理使用当前预测和截至当前已观测的行情，盘中组合统计只在相应历史区间结束后更新。此层单独将 Y3 从 0.31139327 提高到 0.31164268。

## 复现

先按[上一正式版说明](../basket_cash_feedback_v1/README.md)完成上游训练、生成五步盘口版基准预测并核对分数。沿用相同的 Python 环境、CUDA 环境、股票顺序、训练产物目录和基础源码目录。以下命令从仓库根目录执行。若此前自定义了滚动预测目录，向三个命令传入同一个 `--rolling` 路径。日张量缓存的自定义路径通过 `TULIN_CACHE` 传入。

```bash
release_dir="$PWD/project_restart/releases/causal_16d_best"
version_dir="$release_dir/versions/replica_innovation_labels_v1"
prior_src="$release_dir/src"
run_dir="$PWD/local_runs/five_tick_field_curve_v1"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2 POLARS_MAX_THREADS=2

python -u "$version_dir/prepare_base.py" --prior-src "$prior_src" --run-dir "$run_dir"
python -u "$version_dir/train.py" --prior-src "$prior_src" --run-dir "$run_dir"
python -u "$version_dir/replay.py" --prior-src "$prior_src" --run-dir "$run_dir" \
  --report "$run_dir/replica_replay.json"
```

[prepare_base.py](prepare_base.py) 重载上一版各层，用五步盘口版完整日及截断前缀预测重建起点。[train.py](train.py) 在每个时间切点重新拟合，并生成下游所需的滚动预测。已有前置层产物时可用 `--start N`、`--stop N` 指定训练范围。[replay.py](replay.py) 核对输入预测摘要，重载各层参数，独立计算完整日和截断前缀，检查共同区间内预测的最大绝对误差小于 0.0001。报告包含逐日和合并结果。权重、训练数据和预测缓存由运行入口在本地生成。
