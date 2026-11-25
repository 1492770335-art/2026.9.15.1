# 图灵杯 50 股预测模型 · 训练源码

使用 20 个交易日中的前 16 天训练，后 4 天验证。每一帧包含 50 只股票的 100 个字段，每个时点为每只股票输出三个预测。冻结模型在后四天共 936,250 行上的相关系数分别为 **0.50166977、0.39152044、0.30633718**，按 20%、30%、50% 加权得到 **0.37095868**。这是多阶段组合模型在已反复用于比较的验证集上的结果。

## 准备数据与环境

在仓库根目录执行。建议使用 Linux、Python 3.9、支持 CUDA 的 PyTorch 2.5.1。原始比赛 parquet 位于仓库的 Git LFS 中。

```bash
git lfs pull --include="project/project/data/public.parquet"
cd project_restart/releases/causal_16d_best/src
python -m pip install numpy==2.0.1 scipy==1.13.1 scikit-learn==1.6.1 polars==1.36.1 lightgbm==4.6.0 pyarrow tqdm pandas
mkdir -p project/data
ln -s "$(realpath ../../../../project/project/data/public.parquet)" project/data/public.parquet
python -m joint_transformer.prepare_data --data project/data/public.parquet --cache artifacts/joint_transformer/cache
```

缓存按交易日、时间、股票代码排列，形状为 `(时点数, 50, 100)`。日期从 0 开始，训练使用 0–15，验证使用 16–19。数据列 `y0、y1、y2` 顺序对应 Y1、Y2、Y3。

## 从原始数据训练

以下两个命令可以独立运行。第一个训练使用全部 50 只股票的 BookNet，第二个训练组合中的周期回归层。训练结果保存在本机的 `src/artifacts/project_restart_generalization/`。

```bash
python -u train_release.py backbone
python -u train_release.py clock
```

BookNet 同时学习个股收益、市场共同走势和未来盘口状态变化。周期回归以 20 步为一个周期，从前 8 天估计两个交易时段的相位差，再用前 16 天分别拟合市场与个股的响应。Y3 按两个标签的收益尺度承接 Y2 的修正。两步各自产生一个组件，页首指标对应完整组合。[BookNet 训练](src/experiment_phase_future_book_v1.py) · [周期层训练](src/experiment_periodic_book_response_v1.py) · [Y3 修正](src/experiment_periodic_near_carry_v1.py)

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

## Y3 路径编码器

在预测时点取每只股票最近 64 步，将其分为八个连续的 8 步窗口。每个窗口保留 44 个字段的均值。网络先编码每只股票，再结合 50 只股票的平均隐向量，预测个股相对同期市场的收益。第一层分别读取同一段历史的正序视图和反向视图；反向视图同时翻转价格变化的符号。第二层固定这只股票的价格历史，将本股盘口与两组其他股票的盘口配对，估计最近 32 步的价格变化在本股盘口下多提供了多少信息。

第三层学习一条收益曲线。分别以第 0、5、…、40 步为起点，取随后 20 步的相对收益作为九个训练目标。用三个余弦基系数表示这条曲线，最后一层主要取第 21–40 步和第 41–60 步的预测来修正 Y3。训练用的未来收益只出现在监督目标中。预测时，网络可见的数据截止当前时点。编码器按前 4、8、12、16 天扩展训练，后续读出层使用较早日期训练出的模型生成滚动预测。

| 同一切分上的本地版本 | Y3 相关系数 | 加权综合分 |
| --- | ---: | ---: |
| 加入路径编码前 | 0.29403761 | 0.36306992 |
| 过去 64 步的双向视图 | 0.29426158 | 0.36318190 |
| 再区分本股盘口的近期价格响应 | 0.29452040 | 0.36331131 |
| 再加入九个收益窗口的编码器 | 0.29485846 | 0.36348034 |
| 再按个股未来波动幅度微调 | 0.29513001 | 0.36361611 |
| 再利用两个模型方向一致的部分 | 0.29527809 | 0.36369015 |
| 再区分预计与意外的价格变化 | 0.29535001 | 0.36372611 |
| 再区分未兑现、反向、超额三类变化 | 0.29559385 | 0.36384803 |
| 再用同向超额变化修正市场整体预测 | 0.29567527 | 0.36388874 |
| 再根据两个组合的方向一致程度调整幅度 | 0.29570038 | 0.36390130 |
| 用另一组股票估计相同短期预测下的远期预测 | 0.29570666 | 0.36390444 |
| 根据最近 3 次和 12 次价格变化校准 Y3 | 0.29575732 | 0.36392977 |
| 加入价格变化前后实际观测到的预测修订 | 0.29591260 | 0.36400741 |
| 对齐本股价格变化前后预测的到期时点 | 0.29605525 | 0.36407873 |
| 比较同侧同一价位的两次观察 | **0.29619360** | **0.36414791** |

表中各版的 Y1、Y2 预测相同。最终版本在第 17–20 天的单日综合分分别为 0.408872、0.310867、0.352916、0.374448。相对上一次正式发布的 0.36240663，累计提高 0.00174128。后四天曾多次用于比较模型，泛化能力仍须在新日期检验。

### 单独训练和使用九窗口编码器

在准备好每日缓存的机器上进入 `src`。下列命令使用 CUDA 训练 6 轮，输出位置可自行指定。

```bash
python -u train_release.py path-pretrain --output /tmp/dense16.npz
```

以下示例为第 17 天的前 64 个时点生成预测。结果形状为 `(64, 50, 2)`，最后一维依次为第 21–40 步和第 41–60 步的相对收益信号。它们是最终 Y3 预测的两个输入。

```python
import numpy as np
from experiment_causal_cross_horizon_ridge_v1 import CACHE
from experiment_y3_expanding_path_complement_v1 import encode

raw = np.load(CACHE / 'day_16_x.npy', mmap_mode='r')
with np.load('/tmp/dense16.npz') as weights:
    forecasts = encode(raw[:64], weights, only_dense=True)[:, :, 4:6]
print(forecasts.shape)
```

### 训练与回放完整组合

复核 0.36369015 需要上游专家权重、训练期滚动预测和前一层的回放记录。单组件训练可直接使用每日缓存。完整组合训练须在本地重建上游产物。上一发布版之后，依次加入开盘相位、同股与跨股盘口修正、Y3 的市场与板块因子、订单压力以及个股相对排序。各层在 `layer()` 中显式调用前一层。逐层使用固定参数生成第 17–20 天的预测，才可以对照上表。早期的会话组合使用原始 parquet；请保留准备步骤中的 `project/data/public.parquet` 链接。[完整组合入口](src/released_model.py) · [排序模型](src/experiment_y3_pair_crossing_premia_v1.py) · [训练期滚动预测](src/y3_crossing_validation.py)

在排序模型及其上游已经训练完成的前提下，依次执行以下命令。第一个命令在第 5–8 天生成训练读出层所用的历史基线。第二个命令从较早模型重建第 9–16 天的滚动预测。后面的训练脚本会验证训练期切分，并重新加载参数复核后四天结果。

```bash
python -u y3_early_score_neighborhoods.py
python -u y3_crossing_validation.py
python -u experiment_y3_two_sided_path_encoder_v1.py
python -u experiment_y3_path_price_context_v1.py
python -u cache_y3_context_reference.py
python -u experiment_y3_expanding_path_complement_v1.py
python -u y3_tail_specialist_eval.py experiment_y3_expanding_path_complement_v1
python -u verify_y3_expanding_path_readout.py
```

九窗口编码器与之前各层组合的得分在本机的 `artifacts/project_restart_generalization/y3_expanding_path_complement_v1/` 下。对照 `raw_replay.json`、`reload_verification.json` 和 `full_chain_verification/composed_replay.json` 的逐日分数及汇总相关系数。后四天总计 936,250 行，计分先合并四天的样本，再对三个相关系数加权。按收益区间检查 Y3 的命令如下。

```bash
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_expanding_path_complement_v1
```

### 微调个股的方向和波动幅度

新增网络从九窗口路径编码器的权重开始训练。每个 20 步窗口同时预测标准化方向和该股的波动幅度。波动监督值由窗口内的逐步价格变化均方根计算。模型在训练时学习标准化方向、实际相对收益和对数波动幅度；在推理时使用截至当前的盘口估计两者，再将方向乘以本股的预测幅度。方向相同的两只股票因此可以获得不同的修正幅度。训练日期内的四个滚动切分用来选择读出方式。

准备好每日缓存后，下面的命令可以单独训练这个组件。程序会使用本地已有的九窗口编码器，或从每日缓存重建所需权重。

```bash
python -u train_release.py path-scale --output /tmp/y3_scale16.npz
```

组件输出可以从 `experiment_y3_scale_conditioned_path_v1.encode(raw, weights)` 读取。第 4、5 列特征（从零编号）是按本股预测幅度调整后的方向，分别对应第 21–40 和第 41–60 步。要得到波动幅度模型的 0.36361611 分，先完成上一节的所有训练及回放，然后执行：

```bash
python -u y3_expanding_path_references.py
python -u experiment_y3_scale_conditioned_path_v1.py
python -u y3_tail_specialist_eval.py experiment_y3_scale_conditioned_path_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_scale_conditioned_path_v1
```

波动幅度模型的报告保存在 `artifacts/project_restart_generalization/y3_scale_conditioned_path_v1/`。首次回放和重载回放使用同一组冻结参数；`full_chain_verification/composed_replay.json` 另外从上游预测重新组合所有修正。[波动预测与微调源码](src/experiment_y3_scale_conditioned_path_v1.py) · [训练期滚动校验](src/y3_expanding_path_references.py)

固定读出层系数后，将本股预测幅度替换为其他股票的幅度，综合分从 0.36361611 降到 0.36354507；只用微调后的方向得到 0.36358525。另一个事后对照保留新的读出系数、换回旧路径信号，得到 0.36363116。因此这次实验确认了整套配置的增益，独立归因给网络微调的证据仍有限。该对照只用于分析；版本按训练期滚动验证选定。分块重采样估计的波动幅度模型相对九窗口编码器的综合分增益 95% 区间为 −0.000237 至 +0.000635。

```bash
python -u audit_y3_scale_conditioned_path.py
```

### 用两个编码器一致的方向校准 Y3

对于第 21–40 步和第 41–60 步收益窗口，分别比较原始路径编码器与波动幅度模型的输出。若两者都预测上涨或都预测下跌，取绝对值较小的一项并保留方向；若方向相反，该项贡献为零。再减去同帧 50 股均值，用训练期估计的读出系数修正 Y3。这个规则作用于这项附加信号，整体预测仍包含前面各层。训练期与其他融合方式同场比较，四个滚动切分选中了该规则。

在已经完成波动幅度模型训练的 `src` 目录运行：

```bash
python -u y3_scale_conditioned_references.py
python -u experiment_y3_path_magnitude_consensus_v1.py
python -u y3_tail_specialist_eval.py experiment_y3_path_magnitude_consensus_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_path_magnitude_consensus_v1
```

最终报告在 `artifacts/project_restart_generalization/y3_path_magnitude_consensus_v1/`。该层相对波动幅度模型的综合分增益为 +0.00007404，第 17 天单日小幅下降，其余三天上升。此次比较在 120 步分块重采样下的 95% 增益区间为 −0.000174 至 +0.000407。可用 `audit_y3_scale_training_transfer.py` 核查前 16 天滚动训练时替换方向或幅度的影响。[方向一致性训练](src/experiment_y3_path_magnitude_consensus_v1.py) · [滚动参考预测](src/y3_scale_conditioned_references.py)

版本发布按同一切分、全量回放和重载后的未舍入分数判断。自上次正式发布累计提高至少 0.001 时，提交训练代码和使用说明。

## 预计价格变化与实际成交方向

在前四天的盘口上训练一个岭回归模型。它在时点 t−1 预测每只股票下一步相对同期股票的价格变化。到 t 时，系统已经观察到实际变化，可以计算实际与预计的差。第一层在相同盘口下分别编码原始历史以及将这部分差异反向后的历史，再用两种编码的差修正 Y3。

第二层将一次价格变化的预测误差分成三项。例如预计上涨 2 个单位，实际上涨 1 个单位，未兑现的部分为 −1。若实际下跌 1 个单位，未兑现部分为 −2，反向部分为 −1。若实际上涨 3 个单位，超额部分为 +1。每项单独进入路径编码器，训练期的滚动预测用于估计读出系数。该层冻结 Y1 和 Y2。

进入前文的 `src` 目录，可以单独从每日特征训练单步价格模型。它在第 5–16 天的日均截面去均值相关系数约为 0.384。

```bash
OPENBLAS_NUM_THREADS=1 python -u train_release.py price-teacher --output /tmp/price_teacher.npz
```

完整模型需要前文各阶段的训练产物。完成方向一致性训练后按顺序运行以下命令。最后的检查分别重载上游各层回放完整四天、评估各收益区间，以及在训练期滚动预测上检验三类价格误差。

```bash
python -u y3_consensus_references.py
python -u experiment_y3_expected_price_paths_v1.py
python -u verify_y3_surprise_inference.py
python -u y3_surprise_references.py
python -u experiment_y3_unfulfilled_price_moves_v1.py
python -u y3_tail_specialist_eval.py experiment_y3_unfulfilled_price_moves_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_unfulfilled_price_moves_v1
python -u audit_y3_price_error_types.py
```

区分三类价格变化后，相对上一层的综合分增加 0.00012192，第 18 和 20 天改善，第 17 和 19 天略降。对验证集按 120 步分块重采样，综合分增益的 95% 区间约为 −0.000094 至 +0.000415。后四天反复用于模型比较，该区间也未计入多次尝试造成的选择偏差。

前 16 天内的消融实验固定每折的系数，在第 9–16 天的历史预测上把反向变化的信号置零。两组参考预测下，Y3 相关系数增益分别由 +0.000176、+0.000122 变为 −0.000184、−0.000216。这表明当前读出层依赖反向变化。后四天的独立收益区间检查中，第 41–60 步的个股相对收益相关系数提高约 0.000303。

## 市场整体的 Y3 修正

把 50 只股票按固定的代码顺序交替分成两个各含 25 股的组合。分别计算每只股票价格变化与上一帧盘口预期同向、且实际幅度超出预期的部分，保留其正负号。在每个组合内对经过两种强度压缩的数值求均值，然后分别汇总最近 4、16、48 步。用一个组合的特征拟合另一个组合的市场平均 Y3 误差，交换两组再拟合，合并两组提供的训练矩。滚动训练同时控制已有的三个市场预测和近期价格变化。推理时将两个组合给出的共同修正取平均，加到全部 50 只股票的 Y3 上。

先完成上一节训练，再在 `src` 中运行。第一条命令重建前版在训练日期内的滚动预测；第二条选择读出方式、拟合参数并在四天验证集评估；第三条用缓存的基础预测和原始盘口特征独立回放各层修正。第四条和第五条用于检查收益区间及复核训练期四个切分。

```bash
python -u y3_unfulfilled_references.py
python -u experiment_y3_market_reversal_flow_v1.py
python -u y3_tail_specialist_eval.py experiment_y3_market_reversal_flow_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_market_reversal_flow_v1
python -u y3_market_flow_references.py
```

相对三类价格变化模型，Y3 提高 0.00008142，综合分提高 0.00004071。第 17 天下滑，随后三天改善。按 120 步分块重采样，综合分增益的 95% 区间约为 −0.000059 至 +0.000158。共同修正保留了同一时点 50 股的相对排序。[市场修正训练与验证](src/experiment_y3_market_reversal_flow_v1.py)

### 根据两个组合的一致程度调整幅度

设两组股票对同向超额变化分别给出有符号的数值 A 和 B。程序计算 `2*A*B/(A*A+B*B+1e-8)`，再将上一节的市场修正乘以 `1+0.4*一致程度`。两组同时看到同方向变化时提高修正幅度，看到相反方向变化时降低幅度。使用上一层的冻结参数和前 16 天内滚动生成的两组预测，选择这一计算方式。Y1 和 Y2 保持冻结。

完成上一节训练及滚动预测重建后运行以下命令。最后一条命令在训练期改变股票分组，检查增益对初始分组的敏感程度。

```bash
python -u experiment_y3_overshoot_basket_agreement_v1.py
python -u verify_y3_market_gate_replay.py experiment_y3_overshoot_basket_agreement_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_overshoot_basket_agreement_v1
python -u y3_overshoot_agreement_references.py
python -u audit_y3_market_basket_partitions.py
```

这一层相对统一市场修正多得 0.00001256。第 9–16 天的第一组滚动参考预测上，原始分组的 Y3 增益为 0.00003144，更换为八种预先生成的随机分组后，增益在 0.00003776–0.00006278 之间。后四天的分块重采样估计给出 −0.000010 至 +0.000040 的综合分增益区间。验证日期已经用于多轮模型比较。[市场一致性校准](src/experiment_y3_overshoot_basket_agreement_v1.py) · [分组敏感性检查](src/audit_y3_market_basket_partitions.py)

可用以下命令同时评估两层市场修正对近端、远端市场收益的影响，并生成供下一轮试验使用的本地冻结预测缓存。每个交易日的缓存保存完整输入与截断输入的预测，读取时校验输入和预测的摘要。

```bash
python -u audit_y3_market_return_windows.py
python -u cache_y3_market_agreement_reference.py
```

两层合用时，市场 Y3 相关系数从 0.392213 提高到 0.392573。用同一条后 40 步市场预测分别对照两个互不重叠的 20 步收益，第 21–40 步的相关系数从 0.219525 提高到 0.220395，第 41–60 步从 0.112028 变为 0.112000。跨层合计的综合分增益为 0.00005327，分块重采样 95% 区间为 −0.000062 至 +0.000193。

在训练期第 9–16 天，保持超额变化的计算规则相同，使用各股票自身预期时，第一组滚动参考预测上的 Y3 增益为 0.00008552。将预期换给其他股票后重新拟合得到 −0.00000139，将预期延迟 16 步后重新拟合得到 +0.00000881。以下命令复核该对照。

```bash
python -u audit_y3_market_overshoot_placebo.py
```

## 用另一组股票估计远期预测随短期预测的变化

每个时点先将 Y2、Y3 在 50 股之间去均值，再用 Y3 减去按收益尺度换算的 Y2，得到覆盖第 21–60 步的相对收益预测。按代码顺序将股票交替分成两组，在第一组的 25 只股票上以 Y2 为横轴，用单调回归拟合远期预测，再对第二组的股票按 Y2 插值。两组交换后即可得到全部 50 股的信号。计算只读取当前及此前已有的预测。读出层还包括原有三个预测和当前截面的线性拟合，训练期滚动样本用于估计各项的系数。

完成上一节的训练、滚动预测重建和缓存后，在 `src` 运行以下命令。

```bash
python -u experiment_y3_monotone_horizon_projection_v1.py
python -u verify_y3_monotone_inference.py
python -u y3_tail_specialist_eval.py experiment_y3_monotone_horizon_projection_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_monotone_horizon_projection_v1
python -u y3_monotone_references.py
```

该层在后四天的综合分净增 0.00000314，分块重采样 95% 区间为 −0.00000757 至 +0.00001758。增强后的 Y3 对第 41–60 步相对收益的相关系数从 0.01789074 提高到 0.01790141。这一变化很小。首次回放、参数重载和独立组合回放的报告保存在本机 `artifacts/project_restart_generalization/y3_monotone_horizon_projection_v1/`。[单调回归及训练入口](src/experiment_y3_monotone_horizon_projection_v1.py)

在第 9–16 天固定每一折已拟合的读出系数，把另一组合估计的单调曲线换成线性曲线，第一组滚动参考预测上的 Y3 增益从约 −0.00000009 变为 +0.00000296，第二组则从 +0.00000186 变为 −0.00000043。两组对非线性相对线性的优势没有一致结论。将该曲线分配给错误的股票，两组相对各自基线的增益都为负。可在 `src` 运行 `python -u audit_y3_monotone_assignment.py` 复核。

## 用价格变化前后的预测修订校准 Y3

这一层保留 Y1、Y2 的输出，只调整 Y3。先减去 50 股的同期预测均值，把 Y2 换算到 Y3 的收益尺度，再从 Y3 中减去这个近期分量，得到较远期的预测分量。分别计算两个分量相对于上一帧的变化量。每只股票用自己的价格变化来划分历史，保留最近 3 次和 12 次变化的预测修订，并用固定 9 步和 36 步窗口作对照。第二阶段分别记录价格变化前一帧、当帧和后一帧的修订。前一帧的修订在价格变化得到确认时才归入该事件，后一帧的修订从随后一帧实际到达时起才可用。两个阶段均按交易日重置历史，使用训练期内的滚动预测选取读出方式。

接续上节的历史预测，在同一目录运行：

```bash
python -u experiment_y3_quiet_forecast_revisions_v1.py
python -u cache_y3_price_revision_reference.py
python -u y3_price_revision_references.py
python -u audit_y3_revision_timing.py
python -u experiment_y3_event_forecast_response_shape_v1.py
python -u y3_tail_specialist_eval.py experiment_y3_event_forecast_response_shape_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_event_forecast_response_shape_v1
python -u audit_y3_event_response_contributions.py
python -u y3_event_response_references.py
```

两个阶段合计使后四天的加权分提高 0.00010297。逐日增量为 0.00008631、0.00015484、0.00009215 和 0.00008697。配对时间分块重采样的 95% 区间为 +0.00004092 至 +0.00017961，只描述这四天内的波动。以第 41–60 步的个股相对收益单独评估，相关系数依次为 0.01790141、0.01798067、0.01804206。在训练期两组滚动预测上去掉事件前一帧的修订，第二阶段的 Y3 增益均降至完整配置的约八分之一或更低。模型在推理时只使用已经到达的行情与预测。[事件窗口训练](src/experiment_y3_quiet_forecast_revisions_v1.py) · [前后帧修订训练](src/experiment_y3_event_forecast_response_shape_v1.py) · [修订贡献检验](src/audit_y3_event_response_contributions.py)

## 按相同到期时点比较预测

例如，某股上次价格变化发生在三步前。要考察从现在起第 20 步的价格，就取当时模型向前 23 步的预测，并用这三步已观测到的价格变化把起点换到当前。模型通过 5、20、60 步三个收益端点做分段线性插值，在当前第 5、20、40 步分别比较价格变化前一帧、当帧和当前预测。如果历史预测的覆盖期不足，该项记为零。训练时用前 16 天中的历史滚动预测拟合完整 Y3 的误差。每日重新开始记录事件。

先运行上一节的命令生成参考预测，然后在 `src` 执行下列命令。第一条筛选并拟合这一层，第二条从原始行情组合上游模型，第三条用四个互不重叠的实际收益区间归因，最后一条重建训练期滚动预测。

```bash
python -u experiment_y3_same_deadline_event_curve_v1.py
python -u y3_tail_specialist_eval.py experiment_y3_same_deadline_event_curve_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_same_deadline_event_curve_v1
python -u y3_same_deadline_references.py
```

该层相对上一阶段的加权增益为 +0.00007132，四天中三天提高。配对时间分块重采样的 95% 区间为 −0.00000137 至 +0.00015482。第 41–60 步个股相对收益的相关系数从 0.01804206 升至 0.01815666。所有股票和时点上的官方 Y3 相关系数为 0.29605525。[模型及训练入口](src/experiment_y3_same_deadline_event_curve_v1.py) · [滚动筛选与分段评估](src/y3_full_residual_trial.py)

## 比较同一侧、同一价位的两次观察

原始 `f000` 的非零变化常常是个股价格步长的整数倍。用前四天估计百分比变化与整数价格坐标之间的尺度，推理时每隔 120 步使用截至上一帧的最近至多 960 步重新估计坐标。再从 `f040` 推算中心两侧对称的距离，并用 `f010` 约束坐标的奇偶性。这里的上下两侧是按匿名字段推算出的价位。数据没有提供字段的业务名称。

当某只股票的一侧重新回到相同价格时，取 240 步内该侧上一次处在这个价格时的字段值，与当前值作差。要求两次观察之间至少隔了一帧。每侧使用四个字段，其选择只依赖前四天相应一侧价格变动时字段变化的强弱。将两侧差值在过去 8 步、40 步内分别求和，用前 16 天内的滚动预测训练 Y3 校正。发布模型保存字段编号、价格换算参数和回归系数，推理逐日重置价位历史。[训练与推理](src/experiment_y3_same_quote_memory_v1.py) · [价格换算](src/y3_round_price_coordinates.py) · [两侧字段的训练期对照](src/audit_y3_quote_side_fields.py)

完成上一节后，在同一个 `src` 目录执行以下命令。

```bash
python -u experiment_y3_same_quote_memory_v1.py
python -u verify_y3_quote_memory_inference.py
python -u y3_tail_specialist_eval.py experiment_y3_same_quote_memory_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_same_quote_memory_v1
python -u audit_y3_same_quote_assignment.py
python -u y3_quote_memory_references.py
```

这一层把后四天全量验证的综合分从 0.36407873 提高到 0.36414791，Y1、Y2 的预测逐行一致。第 17–20 天的单日综合分变化分别为 −0.00003732、+0.00032838、−0.00007824、+0.00003093。按日和时间块配对重采样，增益的 95% 区间为 −0.00007147 至 +0.00027402。后四天已用于多次模型比较。

训练期另取同一次回访，在真正匹配的旧观察附近最多前后 6 步寻找不同价位，固定当前时点、字段、股票、事件和模型宽度后重新拟合。在可配对的 850,679 次回访中，两组滚动参考预测的平均 Y3 增益分别从同价位的 +0.0001343、+0.0001118 降到不同价位的 +0.0000496、+0.0000346。[价位配对对照](src/audit_y3_same_quote_assignment.py)

用冻结参数预测时，将同一天分别截断在 9 个位置，所得前缀与完整日预测的最大差异均为零。逐股按时间读取的字典与批量算法在 66,890 个有效的侧别和时点上指向同一条历史观察。第 41–60 步的个股相对收益相关系数从 0.01815666 提高到 0.01830680。[冻结推理检查](src/verify_y3_quote_memory_inference.py)

## 用最近的价位回访事件预测 Y3

对每只股票分别跟踪上下两个报价，每侧保留最近四次同价位回访事件。每个事件记录四个字段相对上次到访的变化、两次到访的间隔，以及本次到访距当前的时间。超过 240 步的事件失效。模型用当前三个期限的预测计算事件权重，按侧汇总，再预测个股相对 50 股均值的 Y3 修正量。训练时同时拟合完整 Y3 残差和后 40 步残差。前 16 天内的滚动筛选选中了完整 Y3 输出头，修正强度固定为 0.24。GPU 用于训练，冻结推理在 CPU 上执行。[事件编码、训练与推理](src/experiment_y3_quote_event_tokens_v1.py)

完成上一节的滚动参考预测重建后，在同一个 `src` 目录执行以下命令。

```bash
python -u experiment_y3_quote_event_tokens_v1.py
python -u verify_y3_quote_event_tokens.py
python -u y3_tail_specialist_eval.py experiment_y3_quote_event_tokens_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_quote_event_tokens_v1
python -u y3_quote_token_references.py
python -u audit_y3_event_token_horizons.py
```

后四天全量回放的综合分从 0.36414791 升至 0.36415697。四天的单日变化依次为 −0.00001661、+0.00020484、−0.00013522、−0.00005853。时间分块配对重采样的增益 95% 区间为 −0.00014661 至 +0.00017446，跨日增益仍有较大不确定性。冻结推理在 9 个截断位置上得到完全相同的前缀预测，独立的逐股时间顺序实现复核了 6,728 组事件输入。[冻结推理复核](src/verify_y3_quote_event_tokens.py)

在第 9–16 天的四个滚动窗口中，两个参考模型的 Y3 预测与第 21–40、41–60 步个股相对收益的相关均有提高。前 5 步的相关下降。将变化拆为新增预测与收益的协方差、整体预测标准差两个贡献后，两者都解释了前 5 步的下降。后四天 924,250 个具有连续价格记录的时点与股票组合上，第 41–60 步相关从 0.01830680 提高到 0.01859649，第 1–5 步从 0.34538969 降到 0.34426348。[训练期分段归因](src/audit_y3_event_token_horizons.py) · [后四天分段统计](src/audit_y3_pair_return_intervals_v1.py)

## 识别单侧成交落在上一帧报价的事件

用训练数据估计整数价格网格，从匿名字段 `f010` 还原一个随成交更新的价格。第 9–16 天，在只有一侧出现成交的有效样本中，这个价格约有 81.1% 等于上一帧同侧报价，约有 0.8% 等于另一侧报价。它与最后成交价的行为相符。另一个字段 `f011` 与按两侧成交量加权的均价相符。两种解释分别通过交易不活跃时的价格稳定性、两侧都有成交且报价稳定时的量价一致性进行了核对。匿名字段的实际业务名称仍未知。[成交位置核对](src/audit_y3_quote_trade_location.py) · [成交均价核对](src/audit_y3_print_vs_average_price.py)

筛选选中了单侧成交且推算成交价等于上一帧同侧报价的观测。每个事件记录买卖方向、按本股训练期平均量缩放后的带符号成交量。分别在过去 5、20、60、180 步累积这些值，经 50 股截面去均值后拟合 Y3 的剩余误差。第 9–16 天的相同样本上，两套滚动参考模型的平均 Y3 相关增益分别为 +0.000163、+0.000177，将比较价替换成当前报价时为 +0.000117、+0.000119。[事件与预测](src/experiment_y3_inferred_print_impacts_v1.py) · [等价匹配对照](src/audit_y3_print_assignment_controls.py)

完成上一节后，在同一个 `src` 目录执行以下命令。

```bash
python -u experiment_y3_inferred_print_impacts_v1.py
python -u verify_y3_inferred_print_impacts.py
python -u y3_tail_specialist_eval.py experiment_y3_inferred_print_impacts_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_inferred_print_impacts_v1
python -u audit_y3_print_assignment_controls.py
python -u audit_y3_quote_trade_location.py
python -u audit_y3_print_vs_average_price.py
python -u y3_inferred_print_references.py
```

这次后四天全量验证的综合分从 0.36415697 升到 0.36417948。第 17–20 天的单日增益依次为 +0.00002780、+0.00014419、+0.00000216、−0.00009527。增益的配对时间分块重采样 95% 区间为 -0.00014167 至 +0.00025420。后 40 步个股相对收益的相关下降了 0.00023642。整体增益很小，跨日的稳定性仍待验证。

## 估计当前方向的成交还能持续多久

沿用上一层的成交事件。对于每只股票，在最近一笔符合条件的成交后持续跟踪方向，直到出现一笔反方向且同样落在上一帧报价的成交。每个时点记录连续同向成交笔数、相邻成交的平均间隔、距最后一笔的步数及近期成交频率。超过 80 步没有新事件时，该状态失效。这里的“笔数”按比赛帧计数，单帧可能包含多笔原始成交。

用已结束的训练日统计在方向反转前，未来第 1–5、6–20、21–40、41–60 步里预计还会发生多少个同方向的事件。样本较少的组合向仅按近期频率和距上次成交时间划分的组收缩。所有输入侧的运行状态均按时间正向计算。训练中的未来事件只用于构造先前交易日的学习目标。将四个计数预期、当前方向及其与已有短期预测的交互项输入 Y3 校正层。Y1、Y2 的输出逐行保留。[时序状态、目标和拟合过程](src/experiment_y3_trade_spell_survival_v1.py)

完成上一节后，在同一个 `src` 目录执行以下命令。

```bash
python -u experiment_y3_trade_spell_survival_v1.py
python -u verify_y3_trade_spell_inference.py
python -u y3_tail_specialist_eval.py experiment_y3_trade_spell_survival_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_trade_spell_survival_v1
python -u audit_y3_trade_spell_predictions.py
python -u audit_y3_trade_spell_gates.py
python -u audit_y3_release_gain_v1.py
python -u y3_trade_spell_references.py
```

本层将综合分从 0.36417948 提高到 0.36422023。第 17–20 天的单日变化依次为 +0.00000084、+0.00015079、+0.00002657、−0.00000792。相对上一层的配对时间分块重采样 95% 增益区间为 −0.00001884 至 +0.00013085。训练期第 9–16 天另有 429,096 个可用观测。把成交段长度和成交速度纳入对照后，预测未来同向事件数的均方误差在前 5 步下降 2.02%，在第 41–60 步下降 0.13%。这些数字衡量辅助任务的预测能力。[成交数预测对照](src/audit_y3_trade_spell_predictions.py) · [冻结推理与逐步计数检查](src/verify_y3_trade_spell_inference.py)

训练期的等宽对照表明，完整校正在两套滚动预测上的平均 Y3 增益为 +0.000299、+0.000174。仅保留成交段预期和本股短期预测的交互时，增益为 +0.000295、+0.000172；仅保留成交段自身特征时为 +0.000010、+0.000007。在同样允许交互的条件下，只使用时间和近期活跃度得到 +0.000230、+0.000136。观察到的主要作用是按本股成交状态调整短期预测对 Y3 的贡献。冻结校正通过 9 个时间截断和 3,364 组逐时运行状态复核。[等宽交互对照](src/audit_y3_trade_spell_gates.py)

相对远端提交 `2e58a9d` 记录的综合分 0.36240663，累计提高 0.00181360。直接配对相同时点和股票，第 17–20 天的差值分别为 −0.00054343、+0.00406508、+0.00197742、+0.00217893。以日期和 120 步时间块作 1,600 次配对重采样，综合增益 95% 区间为 −0.00161453 至 +0.00711331。四天验证集也已被多次用于模型比较。[与已发布版本的逐行对照](src/audit_y3_release_gain_v1.py)

## 用其他股票的报价变动修正 Y3

对每只股票分别记录三类消息。买一上移或卖一下移时记录新挂单量，买一下移或卖一上移时记录退出最优价的旧挂单量。在报价保持原位时，用挂单量变化加上该侧成交量估计净补单。买卖两侧按相反方向记号，按各股成交规模缩放，再分别汇总过去 5、20、60 步。最优价挂单量由匿名字段 f021、f037、f045 估计，校准使用最早 4 天。第 9–16 天筛出的 21,809 次相邻两价成交中，在成交仅发生于这两个价位的假设下，估计的旧价挂单量与推算的旧价成交量之间的中位相对误差为 0.19%。[挂单量恢复与核验](src/y3_reconstructed_touch_queues.py)

用此前已结束交易日的数据估计每只股票的成交压力与其他股票未来 5 步相对价格变化的关系，在两组交易日上保留方向一致的连接，截取四个主要方向。每只股票从其他 49 股接收三类消息，同股连接为零，每个接收端的权重之和为零。在前 16 天滚动拟合 Y3 剩余误差，最终权重固定。Y1、Y2 完全沿用上一版。[关系估计](src/experiment_y3_transaction_pressure_network_v1.py) · [消息传递与训练](src/experiment_y3_peer_depth_event_causes_v1.py)

第 9–16 天的四个时间折中，两套参考预测的平均 Y3 相关增益分别为 +0.000878、+0.000987。将股票连接随机重新对应后，增益分别为 −0.000004、−0.000003。保留连接、移动报价一律使用固定量，仍有 +0.000859、+0.000963。单独使用报价改善与退让，得到 +0.000886、+0.000996。主要可重复的信号来自哪只股票、哪一侧报价先变动，精确挂单量的额外贡献较小。[等宽对照](src/audit_y3_peer_depth_mechanism.py)

在上一节执行完毕后，于同一 src 目录运行。

```bash
python -u y3_reconstructed_touch_queues.py
python -u experiment_y3_peer_depth_event_causes_v1.py
python -u verify_y3_peer_depth_inference.py
python -u audit_y3_peer_depth_mechanism.py
python -u y3_tail_specialist_eval.py experiment_y3_peer_depth_event_causes_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_peer_depth_event_causes_v1
python -u audit_y3_release_gain_v1.py peer-depth
python -u y3_peer_depth_references.py
```

后四天共 936,250 行，Y3 相关从 0.29633824 提高到 0.29664193，综合分从 0.36422023 提高到 0.36437207。第 17–20 天综合分变化分别为 +0.00006611、+0.00061280、−0.00002231、−0.00016502。相对上一版的配对分块重采样 95% 增益区间为 −0.00024933 至 +0.00066064。相对远端提交 2e58a9d，综合分累计提升 0.00196544，配对重采样 95% 区间为 −0.00159177 至 +0.00761455。验证仅覆盖四天，跨日增益仍不确定。
从后四天取出 924,250 个可计算连续收益的股票时点，Y3 预测与第 6–20、21–40、41–60 步个股相对收益的相关分别提高 0.00053537、0.00024410、0.00015971，第 1–5 步下降 0.00020556。冻结推理在 11 个时间截断位置的结果与整日运行完全相同，另外用逐股逐帧实现检查 811 帧，最大差异为 1.50 × 10⁻⁸。[逐步推理检查](src/verify_y3_peer_depth_inference.py)

## 向其他股票传递同一未来时点的预测修订

源股票的买一价或卖一价变动时，计算它对未来 5、20、40 步价格的最新估计。上一帧距离同一个未来时点多一步，将已有的 5、20、60 步预测转换到对数价格尺度，并按期限线性插值。两次估计相减时计入这一帧实际发生的价格变化。沿上一节估计的股票间有向连接传递三组修订，在目标股票汇总最近 5、20、60 步的消息，用前 16 天学习 Y3 修正。源股票的预测与价格均取自在当前时点已经得到的帧。[训练与推理](src/experiment_y3_peer_terminal_price_news_v1.py)

第 9–16 天四个滚动窗口的两套参考预测上，该方法的平均 Y3 相关增益分别为 +0.000947、+0.001104。保留相同数量的参数，仅使用已发生的价格变化时为 +0.000508、+0.000622，仅使用预测修订时为 +0.000208、+0.000219。沿用上一帧的原期限时为 +0.000881、+0.001044。两次预测对齐到同一时点有小幅增益，价格变化和预测修订的组合贡献更大。随机重新对应股票后增益接近零。[等宽对照](src/audit_y3_peer_terminal_news_controls.py)

完成上一节后，在同一 src 目录执行。

```bash
python -u experiment_y3_peer_terminal_price_news_v1.py
python -u verify_y3_peer_terminal_news.py
python -u audit_y3_peer_terminal_news_controls.py
python -u y3_tail_specialist_eval.py experiment_y3_peer_terminal_price_news_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_peer_terminal_price_news_v1
python -u audit_y3_release_gain_v1.py peer-news
python -u y3_peer_terminal_news_references.py
```

第 17–20 天合并计算，Y3 相关从 0.29664193 提高到 0.29691296，综合分从 0.36437207 提高到 0.36450759。单日综合分差值依次为 −0.00028232、+0.00097872、−0.00007990、−0.00011804。相对上一层的配对时间分块重采样 95% 增益区间为 −0.00035726 至 +0.00098339，跨日不确定性较大。11 个时间截断复现了完整输入的前缀预测，另用逐股逐帧实现复算 37,150 个观测，最大误差为 1.50 × 10⁻⁸。[独立核对](src/verify_y3_peer_terminal_news.py)

相对 GitHub 提交 2e58a9d 的 0.36240663，累计提高 0.00210096。四个交易日综合分增益依次为 −0.00075964、+0.00565660、+0.00187521、+0.00189587。对交易日和日内 120 步时间块作 1,600 次配对重采样，累计增益的 95% 区间为 −0.00182646 至 +0.00855048。[与远端版本的配对比较](src/audit_y3_release_gain_v1.py)

对后四天 924,250 个可计算各段相对收益的股票时点，Y3 预测与第 6–20、21–40 步个股相对收益的相关分别增加 0.00062110、0.00021135，与第 1–5、41–60 步的相关分别下降 0.00008340、0.00003940。新层的观测增益主要集中在中段。[分段复核](src/audit_y3_pair_return_intervals_v1.py)

## 联合预测未来成交与价格，校正 Y3

把每帧 50 股的净成交方向和 50 股的单步价格变化组成 100 维状态。买卖同时出现或同时缺席时，该股的净方向为零。价格变化先转到对数收益尺度，再作反双曲正弦变换。用训练日拟合带正则化的线性自回归。模型从过去 1、2–5、6–19、20、21–40 步的状态分别估计当前 100 个量，允许一只股票的成交和价格参与预测另一只股票的下一帧。[状态与拟合](src/y3_joint_order_price_state.py)

从当前已知历史出发，把模型生成的未来成交和价格递归用于下一步预测，累计未来 1–5、6–20、21–40、41–60 步的预测价格变化。推理预先合并递归系数，并在八个历史区间内对系数取平均。对这四段预测取相对 50 股均值的偏离，用滚动参考模型的 Y3 残差拟合最后的线性校正。四个滚动窗口的两套参考预测上，Y3 平均增加 0.000275 和 0.000344。第 9–16 天的八个单日中，第一套参考有六天提高。[训练与冻结推理](src/experiment_y3_joint_order_price_rollout_v1.py)

完成上一节的训练和滚动预测重建后，在同一 src 目录执行。

```bash
python -u experiment_y3_joint_order_price_rollout_v1.py
python -u verify_y3_joint_order_price_rollout.py
python -u audit_y3_joint_rollout_controls.py
python -u y3_tail_specialist_eval.py experiment_y3_joint_order_price_rollout_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_joint_order_price_rollout_v1
python -u audit_y3_release_gain_v1.py order-price
python -u y3_order_price_rollout_references.py
```

后四天 936,250 行，Y3 相关从 0.29691296 提高到 0.29698412，总分从 0.36450759 提高到 0.36454317。单日总分差值依次为 +0.00010487、−0.00003284、+0.00004151、+0.00001371。与上一版的配对分块重采样 95% 增益区间为 −0.00008325 至 +0.00015240，跨日收益仍有不确定性。另在 5,800 个股票时点逐帧递归并独立计算 Y3 校正，最大绝对差为 9.32 × 10⁻⁹，11 次截断历史输入后前缀校正的最大差为 2.10 × 10⁻⁹。[独立递归核验](src/verify_y3_joint_order_price_rollout.py)

在前 16 天内，同样用四段未来收益校正 Y3，仅保留历史状态的一次直接传递，两组滚动参考的平均 Y3 增益为 +0.000078、+0.000127。递归时将尚未发生的成交状态设为零，增益为 +0.000103、+0.000157；每只股票仅使用本股动态，增益为 +0.000097、+0.000219。完整模型得到 +0.000275、+0.000344。这些对照衡量各项机制对现有预测的补充程度。[相同预测窗口的对照](src/audit_y3_joint_rollout_controls.py)

在后四天 924,250 个可计算分段相对收益的股票时点，Y3 预测与第 1–5、6–20、21–40、41–60 步收益的相关变化依次为 −0.00011694、+0.00023015、−0.00005872、+0.00009127。[后四天分段复核](src/audit_y3_pair_return_intervals_v1.py)

相对远端提交 2e58a9d 的 0.36240663，后四天逐行对照的累计综合分增益为 +0.00213654。按日期和日内 120 步时间块进行 1,600 次配对重采样，累计增益的 95% 区间为 −0.00178464 至 +0.00853898。[与远端版本的比较](src/audit_y3_release_gain_v1.py)

## 用跨日重复出现的成交关系校正 Y3

仍使用最近 40 帧中 50 股的净成交方向和价格变化。从最早四天估计各股的均值与尺度，把最近历史汇总为五个区间，共 500 个输入。用已经结束的交易日构造四个未来区间的辅助标签，分别是第 1–5、6–20、21–40、41–60 步每只股票的成交与价格变化之和。各段先按长度平方根缩放，再减去同期 50 股平均。推理时只需当前和以前的帧。[训练与推理](src/experiment_y3_persistent_order_modes_v1.py)

两个分支分别预测四段价格变化。个股分支为每只股票单独从本股 10 个历史量拟合回归。联合分支在两组交错交易日上分别测量 500 个历史量与未来成交的关联，将历史输入按正则化协方差白化。用两组关联矩阵的对称交叉乘积保留至多 16 个正特征值对应的方向，沿这些方向拟合未来四段价格变化。在每只股票上将两组四段预测相加，再用当时已有的基础预测和训练日 Y3 残差拟合最终校正。Y1 和 Y2 沿用上一版。

第 9–16 天有四个向前滚动的两日验证窗口。对相同的四段预测和相同的 Y3 校正，单用个股分支时，两套参考预测的平均 Y3 增益为 +0.000413、+0.000548。加入跨日重复的未来成交方向后为 +0.000770、+0.000974。将其中一组训练日期的未来股票重新对应后为 +0.000385、+0.000541。[个股分支固定时的对照](src/audit_y3_persistent_mode_controls.py)

完成上一节后，在同一 src 目录执行。

```bash
python -u experiment_y3_persistent_order_modes_v1.py
python -u audit_y3_persistent_mode_controls.py
python -u audit_y3_persistent_order_sources.py
python -u audit_y3_persistent_order_refit.py
python -u verify_y3_persistent_order_modes.py
python -u y3_tail_specialist_eval.py experiment_y3_persistent_order_modes_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_persistent_order_modes_v1
python -u audit_y3_release_gain_v1.py persistent-orders
python -u y3_persistent_order_references.py
```

第 17–20 天合并验证，936,250 行的 Y3 相关从 0.29698412 提高到 0.29705537，总分从 0.36454317 提高到 0.36457879。单日综合分变化分别为 +0.00048066、−0.00012248、−0.00016437、−0.00020880。相对上一版本的配对时间分块重采样 95% 总分增益区间为 −0.00030195 至 +0.00039699。此次提高集中在第 17 天。单独逐帧构造输入并重算 6,150 个股票时点的校正，最大绝对误差为 1.17 × 10⁻⁸。12 次截断的前缀校正最大差为 6.52 × 10⁻⁹。[逐帧复核](src/verify_y3_persistent_order_modes.py)

保持个股直接预测相同，在第 9–16 天只保留联合分支中来自其他股票的输入，平均 Y3 增益为 +0.000759、+0.000952；只保留该分支中本股的输入，增益为 +0.000439、+0.000579。将其他股票的贡献统一为平均权重，增益为 +0.000496、+0.000649。原始逐股权重在这段滚动检验中提供了额外信息。[逐股来源对照](src/audit_y3_persistent_order_sources.py)

在后四天 924,250 个可计算分段相对收益的股票时点，Y3 预测与第 1–5、6–20、21–40、41–60 步收益的相关变化依次为 −0.00034723、+0.00015358、+0.00002260、+0.00017245。[后四天分段复核](src/audit_y3_pair_return_intervals_v1.py)

在训练期内额外冻结第 8 天结束时的参数。第 11–16 天，更新历史与未来成交的映射、仍用第 8 天的 Y3 校正权重，平均 Y3 增益为 +0.000985、+0.001239；两部分都逐折重估时为 +0.000829、+0.000976。这个训练期对照提示，映射与校正幅度的更新频率值得分开检验。上述后四天分数使用两部分都更新到第 16 天的冻结版本。[重新拟合对照](src/audit_y3_persistent_order_refit.py)

相对 GitHub 提交 2e58a9d 的 0.36240663，累计综合分增益为 +0.00217216。对四天和日内 120 步时间块进行 1,600 次配对重采样，增益的 95% 区间为 −0.00167317 至 +0.00834011。[逐行对照](src/audit_y3_release_gain_v1.py)

## 预测远期收益的接收股票组合

新增分支输入最近 40 帧的 50 股成交方向与价格变化。将已经结束的训练日期交错分成两组，分别学习历史输入与各股未来市场相对对数收益的关联。把一组日期第 1–5 步的关联与另一组第 21–40 步的关联配对，第 6–20 步与第 41–60 步配对；再交换两组日期。由此得到一个 50 × 50 矩阵，取对称部分至多八个正特征值对应的股票组合。这些组合描述了过去的行情同时能预示哪些股票较早和较晚的价格变化。[训练与推理](src/experiment_y3_receiver_price_portfolios_v1.py)

将对第 21–40、41–60 步各股收益的联合回归预测投影到这些组合，同时从本股的成交方向和价格历史单独拟合两段收益。把两路结果逐股相加，并在前 16 天滚动预测的 Y3 残差上拟合最终线性校正。拟合组合时用到了已结束日期的未来收益，推理时只需要预测起点及以前的数据。Y1 和 Y2 沿用上一版。

第 9–16 天的四个滚动窗口中，两套参考预测的平均 Y3 相关增益为 +0.000218、+0.000235，第一套参考有七天为正。固定同一个个股分支，分别只用个股、加入未经投影的联合预测、加入仅按早期反应选出的组合，增益依次为 +0.000129/+0.000137、+0.000145/+0.000144、+0.000162/+0.000160。将接收股票的组合权重随机换位，增益为 +0.000183/+0.000182。[等宽度对照](src/audit_y3_receiver_portfolio_controls.py)

完成上一节后，在同一 src 目录执行。

```bash
python -u experiment_y3_receiver_price_portfolios_v1.py
python -u audit_y3_receiver_portfolio_controls.py
python -u verify_y3_receiver_portfolios.py
python -u y3_tail_specialist_eval.py experiment_y3_receiver_price_portfolios_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_receiver_price_portfolios_v1
python -u audit_y3_release_gain_v1.py receiver-portfolios
python -u y3_receiver_portfolio_references.py
```

后四天 936,250 行的 Y3 相关从 0.29705537 提高到 0.29721634，综合分从 0.36457879 提高到 0.36465928。第 17–20 天的单日综合分变化依次为 +0.00015125、+0.00016582、+0.00002794、−0.00006956。本层相对上一版的配对分块重采样 95% 综合增益区间为 −0.00004464 至 +0.00019486。独立逐帧重算 6,400 个股票时点，Y3 校正的最大差为 4.07 × 10⁻⁹；13 次截断历史输入后的最大前缀差为 1.96 × 10⁻⁹。[冻结推理复核](src/verify_y3_receiver_portfolios.py)

在后四天 924,250 个可计算分段相对收益的股票时点，Y3 预测与第 1–5、6–20、21–40、41–60 步收益的相关变化依次为 −0.00009891、+0.00009750、+0.00021748、+0.00012649。两个远期区间均有提高。[分段收益复核](src/audit_y3_pair_return_intervals_v1.py)

相对远端提交 2e58a9d 的 0.36240663，累计综合分增益为 +0.00225265。对交易日和日内 120 步时间块进行 1,600 次配对重采样，增益的 95% 区间为 −0.00161338 至 +0.00844805。[与发布版本的逐行比较](src/audit_y3_release_gain_v1.py)

## 按当日已观察到的跨股波动更新 Y3

每只股票沿用上一层从前期日期学到的八个股票组合。在为某只股票构建组合时，把这只股票自己的权重置零，其余 49 只的权重和为零。每 80 帧更新一次：先把最新已结束的、最多 960 帧的价格变化按 8 帧一组汇总，再将组间的协方差与历史协方差合并。历史部分的权重相当于 40 组观测。按个股估计它对八个组合的敏感度，用敏感度相对历史平均的变化，修正原模型对第 21–40 和 41–60 步收益的预测。与本股分支相加后，通过冻结的线性层校正 Y3。推理时不读取当天的标签。[模型源码](src/experiment_y3_realized_recipient_exposures_v1.py)

第 9–16 天按时间推进的四折比较中，模型在两套参考预测上的平均 Y3 相关增益为 +0.000259 和 +0.000276。保持个股分支相同，固定历史敏感度时为 +0.000165 和 +0.000183，把更新错配给其他股票时为 +0.000155 和 +0.000169。用其他股票下一时间段已实现的收益解释个股同期相对收益，是另外一个检验：动态更新的日均样本外 R² 为 −0.01918，固定历史值为 −0.00384。Y3 的微小提升不足以证明模型准确估计了当天的真实跨股关系。

完成上一节后，在同一 src 目录执行。

```bash
python -u experiment_y3_realized_recipient_exposures_v1.py
python -u verify_y3_recipient_exposures.py
python -u y3_tail_specialist_eval.py experiment_y3_realized_recipient_exposures_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_realized_recipient_exposures_v1
python -u audit_y3_release_gain_v1.py recipient-exposures
python -u y3_recipient_exposure_references.py
python -u audit_y3_exposure_persistence.py
python -u audit_y3_exposure_transfer.py
```

后四天 936,250 行的 Y3 相关从 0.29721634 提高到 0.29732437，综合分从 0.36465928 提高到 0.36471329。第 17–20 天的单日综合分变化依次为 +0.00012058、+0.00016951、+0.00001885、−0.00014456。相对上一层的配对分块重采样 95% 综合增益区间为 −0.00010021 至 +0.00018194。在训练期第 16 天单独逐股重算 4,900 个股票时点，Y3 校正的最大绝对差为 2.97 × 10⁻⁹；15 次截断的最大前缀差为 1.98 × 10⁻⁹。[冻结推理复核](src/verify_y3_recipient_exposures.py)

在第 9–16 天冻结每折的 Y3 权重，并把近期敏感度变化的系数设为零、按早期日期下一行情块的重建误差拟合、或设为一。两套参考预测的平均 Y3 相关增益分别为 +0.000202/+0.000218、+0.000208/+0.000224、+0.000259/+0.000276。按行情重建误差拟合得到的系数范围为 0.090–0.110；它对 Y3 的增益小于完整更新。[行情块检验](src/audit_y3_exposure_persistence.py) · [同一预测头对照](src/audit_y3_exposure_transfer.py)

后四天合计 924,250 个可算分段收益的股票时点，Y3 预测与第 1–5、6–20、21–40、41–60 步相对收益的相关变化为 −0.00011509、+0.00011746、+0.00015595、+0.00004510。[分段收益复核](src/audit_y3_pair_return_intervals_v1.py)

相对 GitHub 提交 2e58a9d 的综合分 0.36240663，累计增益为 +0.00230666。四个验证日的综合分均有提高；对交易日和日内 120 步时间块进行 1,600 次配对重采样，累计增益的 95% 区间为 −0.00158164 至 +0.00857586。[远端基线逐行对照](src/audit_y3_release_gain_v1.py)

## 用历史价格穿越时的盘口修正 Y3

对每只股票分别回看最近 32 和 96 步。以窗口起点为零，如果终点高于起点，就把这段净涨幅涉及的每一小段价位分给最后一次向上穿越它的时点。净下跌时对称计算。只保留当时买一、卖一挂单量之差与价格移动同向的部分。两个窗口各提取六个量，分别描述保留下来的价格幅度、穿越在窗口内的位置、挂单量差、成交流向、总挂单量和价差。组合时按穿越的价格幅度加权。窗口在当日及午间分界处截断。线性层结合已有三个预测值，对 Y3 做一次修正。预测一个时点只需截至该时点的盘口。[训练及推理源码](src/experiment_y3_crossed_level_book_v1.py)

沿用前 16 天训练、后 4 天验证。第 9–16 天的四个滚动切分中，本层对两套参考预测的平均 Y3 相关增益分别为 +0.000150 和 +0.000297。在后四天全部 936,250 行上，Y3 相关从 0.29732437 升到 0.29761894，加权综合分从 0.364713293079355 升到 0.3648605767367643。四个验证日的单日综合分变化依次为 +0.00003911、+0.00051140、−0.00010676、+0.00006843。相对上一层的配对分块重采样 95% 综合增益区间为 −0.00012808 至 +0.00044178。第 16 天另用逐步回溯价位的方法独立复算 3,150 个股票时点，Y3 修正的最大绝对差约为 6.03 × 10⁻⁹，7 次截断输入检查的前缀差均为零。[逐股复核](src/verify_y3_crossed_level_book.py)

完成上一节训练并生成滚动参考预测后，在同一 src 目录中执行以下命令。第一个入口筛选模型并执行后四天回放与参数重载，第三个入口复核完整修正链。最后一个命令为后续实验生成仅使用先前训练日期的历史参考预测。

```bash
python -u experiment_y3_crossed_level_book_v1.py
python -u verify_y3_crossed_level_book.py
python -u y3_tail_specialist_eval.py experiment_y3_crossed_level_book_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_crossed_level_book_v1
python -u audit_y3_release_gain_v1.py crossed-levels
python -u y3_crossed_level_references.py
```

在后四天可重建分段收益的 924,250 个股票时点上，Y3 预测与第 1–5、6–20、21–40、41–60 步个股相对收益的相关变化分别为 +0.00021229、+0.00026829、+0.00032505 和 −0.00000204。本层对第 21–40 步的改善最大。[分段收益复核](src/audit_y3_pair_return_intervals_v1.py)

相对上一次 GitHub 正式发布的提交 2e58a9d，综合分从 0.36240663155303376 提升到 0.3648605767367643，累计增加 0.00245394518。四个验证日的综合分均高于该正式版本。按日及日内 120 步时间块配对重采样，累计增益的 95% 区间为 −0.00146238 至 +0.00894560。[与正式版本的逐行比较](src/audit_y3_release_gain_v1.py)

## 未来价格变动幅度与方向的联合预测

在前四天学习 17 个离散字段取值变化与同一时刻价格变动之间的关系。训练辅助预测器时，把未来 60 步拆成三个 20 步区间。区间内每步归入小幅变动、离散字段支持的同向大幅变动或其余大幅变动，每类分别统计绝对变动之和与带方向的净变动。大幅的阈值为该股前四天单步收益均方根的 0.75 倍。分类所用的当步字段仅用于构造历史训练标签。

推理使用截至当前时刻的数值和离散字段、8 步和 64 步历史以及同期截面信息。网络为每个区间的每一类变动预测幅度与方向，按「预测幅度减去训练期平均幅度，再乘预测方向」计算九项修正。用前 16 天的滚动训练数据估计线性系数，修正 Y3。第 9–16 天四个滚动切分中，两套参考预测的平均 Y3 相关增益为 +0.000049 和 +0.000062。[训练及推理源码](src/experiment_y3_impact_mass_forecast_v1.py)

在第 17–20 天全部 936,250 行上，综合分由 0.3648605767367643 升至 0.3649576255947699，Y3 相关系数由 0.29761894 升至 0.29781303。四天综合分变化依次为 −0.00002757、+0.00020464、+0.00012930、+0.00012714。按日及日内时间块重采样的 95% 综合增益区间为 −0.00004178 至 +0.00025892。第 16 天独立重算 3,950 个股票时点，修正的最大绝对误差小于 0.000000007。[独立复核源码](src/verify_y3_impact_mass.py)

完成前文的训练和历史参考预测重建后，在同一源码目录运行以下命令。最终预测入口为 `released_model.py`，完整验证命令会从基础模型开始应用全部修正层。

```bash
python -u experiment_y3_impact_mass_forecast_v1.py
python -u verify_y3_impact_mass.py
python -u y3_tail_specialist_eval.py experiment_y3_impact_mass_forecast_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_impact_mass_forecast_v1
python -u audit_y3_release_gain_v1.py impact-mass
python -u y3_impact_mass_references.py
```

## 价格跳动中获得盘口支持的位置

同一次涨价可能跨过多个价位。用截至该时刻的离散字段变化估计支持的涨幅，只保留与实际移动同向的部分，并以真实涨幅为上限。分别把这个宽度放在跳动的起始侧和终点侧，比较两种放置方式与当前仍保留价位的重叠长度。下跌时反向计算。在 32 步、96 步窗口分别按重叠长度汇总盘口信息，取两种放置方式的差值。滚动筛选在前一阶段的预测上重估系数。[特征计算](src/experiment_y3_implied_crossing_depth_v1.py) · [组合训练与推理](src/experiment_y3_clock_and_crossing_depth_v1.py)

前 16 天训练后，第 17–20 天的全部 936,250 行上，Y3 相关由 0.29781303 升至 0.29802075，加权分由 0.3649576255947699 升至 0.36506148160353097。四天相对上一层的加权分变化依次为 +0.00002662、+0.00028398、+0.00007741、+0.00002012。这一层的配对分块重采样 95% 增益区间为 −0.00001261 至 +0.00026359。两次修正合计提高 0.00020090。

在上一节步骤完成后运行。完整评分入口是下面的 `y3_tail_specialist_eval.py` 命令，预测接口是 `released_model.SessionRefinement.predict`。

```bash
python -u experiment_y3_clock_and_crossing_depth_v1.py
python -u verify_y3_implied_crossing_depth.py experiment_y3_clock_and_crossing_depth_v1
python -u y3_tail_specialist_eval.py experiment_y3_clock_and_crossing_depth_v1
python -u audit_y3_pair_return_intervals_v1.py experiment_y3_clock_and_crossing_depth_v1
python -u audit_y3_release_gain_v1.py clock-depth
python -u y3_clock_and_depth_references.py
```

两层合计比 GitHub 上一个正式版本（提交 `2e58a9d`，同一切分得分 0.36240663155303376）提高 0.00265485005。与本轮起点 0.3648605767367643 相比，第 17 天变化约 −0.00000095，其余三天分别提高 0.00048863、0.00020672、0.00014726。验证日期已用于多次版本比较。

训练期另将前后两层的 Y3 修正逐行比较。第 9–16 天两套滚动参考上的修正量相关系数分别为 −0.0179、−0.0168。当前盘口被判为高活动的股票时点占 14.27%；按时点开启或关闭修正，该组对两层增益均有贡献。可在完成历史预测重建后运行 `python -u audit_y3_impact_gain_by_state_v1.py` 复核。[分状态对照](src/audit_y3_impact_gain_by_state_v1.py)

## 三个目标分别利用 Y3 模型各阶段的输出

模型保留 31 个阶段对 Y3 预测的逐次修正。每个时点将各阶段的修正拆为 50 股平均值和个股相对平均值。三个输出头分别用这两类信息调整各自的预测。输出头的选择使用前 16 天内按日期滚动生成的预测。运行完上一节的模型后，在同一个 src 目录执行以下命令。

```bash
python -u experiment_y3_joint_stage_readout_v1.py
python -u experiment_joint_horizon_stage_transfer_v1.py
python -u verify_joint_stage_readouts.py
python -u joint_horizon_stage_references.py
python -u audit_y3_release_gain_v1.py joint-stage-readout
```

第 17 至 20 天合计 936,250 行。四天合并后按三个目标相关系数加权，综合分为 0.36575300501109975。相比此前本地的 0.36506148160353097 提升 0.00069152341，相比前一正式版本 2e58a9d 的 0.36240663155303376 提升 0.00334637346。相对本地上一版，四个验证日分别变化 −0.000563、+0.003601、−0.000257、−0.000189，提升集中在第 18 天。验证日期已多次用于架构比较，在新交易日的收益仍待独立验证。

## 早期盘口模型的分目标预测头

这一版将 11 个中间输出提供给三个目标。其中 5 个输出描述个股盘口状态随时间的变化，另外 6 个描述订单压力、活跃度与二者的先后关系。先用已过去的训练日拟合盘口模型，再使用它对随后交易日的预测训练每个目标的输出头。前 16 天内按日期滚动选择后，Y1 沿用原预测，Y2 使用 11 个信号，Y3 使用盘口状态变化和有顺序的订单压力信号。

在上一节完成后，从同一 src 目录执行。

```bash
python -u experiment_joint_early_book_transfer_v1.py
python -u verify_joint_early_book_v1.py
python -u joint_early_book_references.py
python -u audit_y3_release_gain_v1.py early-book
python -u audit_y3_release_gain_v1.py early-book-increment
python -u audit_early_book_horizon_gain_v1.py
```

前 16 天训练、后 4 天全量验证，综合分为 0.3658582408460957。相比本地上一版 0.36575300501109975 提升 0.000105235835，其中 Y3 相关系数提升 0.000192395611。相对本地上一版，四个验证日的综合分分别变化 −0.000235、+0.000376、+0.000178、+0.000183。相比前一正式版本 2e58a9d 的 0.36240663155303376，累计提升 0.003451609293。验证集已多次用于模型比较。

仅看未来 60 步在同一交易时段、数据完整的 912,250 行，Y3 相关系数提升 0.000191。按实际价格变化拆分贡献，第 1–20 步为 −0.000097，第 21–40 步为 +0.000162，第 41–60 步为 +0.000124，复利与标签差异的剩余项约 +0.000001。相对本地上一版，按交易日与 120 步区块配对重采样得到的综合分增量 95% 区间为 [−0.000207, +0.000455]。

## 压力与交易活跃度的时间间隔

模型分别估计两段时间的作用。一段是压力变化与活跃度变化之间的间隔，另一段是这两次变化发生后又过了多久。两段都使用 4、16、64 步指数平滑尺度，组合成 9 种情况。模型比较压力先出现和活跃度先出现的响应，分别估计未来第 1–20、21–40、41–60 步的个股相对收益。使用过去训练日期得到的预测为三个目标单独选择输出权重，Y2 和 Y3 采用新信号。每个时点使用该时点及此前已观察到的输入。

完成上一节后，在同一 src 目录依次运行。

```bash
python -u experiment_y3_independent_order_clocks_v1.py
python -u y3_order_clock_references.py
python -u verify_order_clock_readouts.py
python -u experiment_joint_order_clock_calibration_v1.py
python -u joint_order_clock_references.py
python -u joint_order_clock_references.py verify
python -u audit_y3_release_gain_v1.py joint-order-clocks-increment
python -u audit_y3_release_gain_v1.py joint-order-clocks
```

前 16 天训练，后 4 天共 936,250 行全量验证，分数为 0.3663383614726957。上一节模型为 0.3658582408460957，本次增量 0.0004801206266000。Y3 相关系数由 0.2986922089622486 升到 0.29961114885696055。第 17–20 天各自的综合分为 0.40916510、0.31901305、0.35383088、0.37330186。前三天较上一节提高，第 20 天小幅回落。相对前一正式版本 2e58a9d（0.36240663155303376）累计提高 0.00393172992。以上交易日曾多次用于候选模型比较。

## 按收益幅度训练 Y3 输出头

对每个 20 步区间，先把个股相对收益除以滚动风险估计。按绝对值 0.6 和 1.8 两个阈值，将标准化收益拆为小幅、中等和尾部三个连续部分。中等部分是将数值截断至 ±1.8 后，减去截断至 ±0.6 的部分。训练期按日期滚动比较后保留中等部分，由模型预测，再用当前已知的风险尺度还原。每个时点的风险尺度来自最近最多 140 步的已观察价格变化，以及前四个训练日估计的个股方差。

完成上一节后，在同一 src 目录执行。

```bash
python -u experiment_y3_order_clock_tail_slices_v1.py
python -u y3_medium_order_clock_references.py verify
python -u y3_medium_order_clock_references.py
python -u audit_y3_release_gain_v1.py medium-order-clocks-increment
python -u audit_y3_release_gain_v1.py medium-order-clocks
python -u audit_early_book_horizon_gain_v1.py medium-order-clocks
```

前 16 天训练，后 4 天共 936,250 行全量回放，参数重载后综合分为 0.36639241063377814。相对本轮开始时的 0.3658582408460957 提高 0.00053416979，Y3 由 0.29869221 提高到 0.29971925。第 17–20 天的综合分变化依次为 +0.000449、+0.001081、+0.001279、−0.000639。配对重采样得到的综合分增量 95% 区间为 [−0.000435, +0.001575]。在 912,250 行完整的同一交易时段 60 步路径中，对 Y3 增量的贡献依次为第 1–20 步 −0.000676、第 21–40 步 +0.000875、第 41–60 步 +0.000829，复利与标签差异剩余项约 +0.000008。相对前一正式版本 2e58a9d 的累计综合分提升为 0.00398577908。

## 按盘口事件发生后的时间预测 Y3

从当前和上一帧的买一卖一报价、队列量及推断的主动成交量，构造买卖压力、成交与补单特征。分别统计压力先于活跃度、活跃度先于压力两种情况。第一层从两者之差预测未来第 21–40 和 41–60 步的相对收益。第二层分别保留 4、16、64 步指数平滑输出，学习事件发生后不同时间尺度对三个区段的贡献。各交易时段开始时重置滤波器。训练期按日期滚动拟合区段预测和读出权重。

完成上一节后，在同一 src 目录执行。

```bash
python -u experiment_y3_touch_flow_clocks_v1.py
python -u y3_touch_clock_references.py verify
python -u y3_touch_clock_references.py
python -u experiment_y3_touch_delay_readout_v1.py
python -u y3_touch_age_readout_references.py verify
python -u y3_touch_age_readout_references.py
```

前 16 天训练、后 4 天共 936,250 行验证。加入盘口事件得到 0.36640561620292156，按事件年龄分别估计响应后得到 0.3664418997005491。参数重载复算一致。后一项的综合分增量为 0.0000362835，第 17 天回落，后 3 天提升。按 120 步分块并对日期重采样，增量 95% 区间为 [−0.000087, +0.000173]。相比 GitHub 的 2e58a9d 版本累计提高 0.00403526815。

### 三个目标的独立校准

沿用上述按时间尺度分开的响应，在前 16 天内用滚动预测分别训练三个目标的校准系数。训练期只有 Y2 选择了跨日期方向一致的校准，Y1 和 Y3 采用上一阶段输出。训练与复算从 src 目录运行。

```bash
python -u experiment_joint_touch_age_calibration_v1.py
python -u joint_touch_age_references.py verify
python -u joint_touch_age_references.py
```

全量回放和参数重载均得到 0.3664438015782361。相对上一节提高 0.00000190188。四天逐日综合分依次为 0.40914806、0.31930045、0.35406243、0.37323715。与 0.36640561620292156 相比，本轮累计增量为 0.00003818538，按日期和 120 步数据块配对重采样，95% 区间为 [−0.000098, +0.000183]。同一验证集曾多次用于开发比较。相比远端 2e58a9d 的累计增量为 0.00403717003。

## 按股票共同波动分解盘口响应

用前四个交易日的 20 步收益学习 50 股的共同波动方向。先移除 50 股等权共同走势，再把剩余方向按特征值分成 4、12、33 个方向三组。对每组分别估计盘口压力与交易活跃度先后顺序对未来三个 20 步区间的贡献。先扣除线性盘口压力能够解释的部分，再把各组预测映射回 50 只股票。最终只更新 Y3。分组、盘口模型与输出权重均在训练日期中拟合。

完成上一节后，在 src 目录执行。

```bash
python -u experiment_y3_factor_order_response_v1.py
python -u y3_factor_order_references.py verify
python -u y3_factor_order_references.py
```

后四天 936,250 行全量验证、参数重载综合分均为 0.36645525814820423，Y3 从 0.29981822531266733 升至 0.2998411384526035。第 17–20 天综合分分别为 0.40908203、0.31934969、0.35413760、0.37324657。相对前一节提高 0.00001145657。相对远端发布版 2e58a9d 的 0.36240663155303376 累计提高 0.00404862660。

## 用已结算的盘中收益更新股票之间的共同波动关系

先将共同波动的三组方向分别用于三个目标，在训练期按日期滚动选择输出权重，得到 0.36650082772311987。接下来以训练日的收益相关矩阵为起点，在盘中每 160 步更新一次。更新时只取此前已经走完的 20 步区间，奇数和偶数区间分别估计股票分组，保留两组估计共同支持的成分。训练日期也按奇偶分开估计股票的分组概率，以平滑训练日较少时不稳定的分组。新层用这些权重将盘口先后顺序的影响分配回每只股票并更新 Y3。

完成上一节后，在 src 目录执行。训练入口从前 16 天估计参数，后 4 天仅用于评估。所有新增推断权重由以下步骤生成。

```bash
python -u experiment_joint_factor_order_calibration_v1.py
python -u joint_factor_order_references.py verify
python -u joint_factor_order_references.py
python -u experiment_y3_live_risk_order_v1.py
python -u y3_live_risk_order_references.py verify
python -u y3_live_risk_order_references.py
```

后 4 天 936,250 行全量回放和权重重载均为 0.36653614387371136。相对 0.36645525814820423 提高 0.00008088573。第 17–20 天的综合分变化分别为 −0.00027512、+0.00032927、+0.00038378、−0.00002706。按日期和 120 步块配对重采样，增量 95% 区间为 [−0.000272, +0.000429]。相比 GitHub 版本 2e58a9d 的累计综合分增量为 0.00412951232。

### 分目标校准盘中共同波动信号

在前 16 天的滚动预测中，Y1 和 Y2 选择了按日期斜率中位数缩放的调整。Y3 沿用已经完成盘中更新的预测。运行入口如下。

```bash
python -u experiment_joint_intraday_risk_calibration_v1.py
python -u joint_intraday_risk_references.py verify
python -u joint_intraday_risk_references.py
```

前 16 天拟合，后 4 天 936,250 行全量回放及重新载入参数的综合分均为 0.3665653878000195。第 17–20 天分别为 0.40873324、0.31981935、0.35469382、0.37311869。相对本轮起点 0.36645525814820423 提高 0.00011012965，配对重采样增量的 95% 区间为 [−0.000383, +0.000622]。相对 2e58a9d 的累计提升为 0.00415875625。

### 组合各阶段的预测并学习相近股票的收益差

先把各预测阶段的增量按股票共同波动方向分组，用训练期的滚动预测分别校准 Y2 和 Y3。再用前四个训练日的股票收益相关性建立分组树，训练神经网络预测同一分支两侧的收益差。训练期选出的 Y3 调整来自小分支上联合编码相对独立编码的预测增量。完成上一节后在 src 目录依次运行。

```bash
python -u experiment_joint_factor_stage_stack_v1.py
python -u joint_factor_stage_references.py verify
python -u joint_factor_stage_references.py
python -u experiment_y3_sibling_spread_network_v1.py
python -u y3_sibling_spread_references.py verify
python -u y3_sibling_spread_references.py
```

前 16 天训练，后 4 天 936,250 行全量回放和参数重载综合分均为 0.36661275891244105。分阶段校准后的综合分为 0.366605111449492，收益差网络进一步将 Y3 从 0.2999175740 提高至 0.2999328689。最后这一步在第 17、20 天有提升，在第 18、19 天有回落。相对上次正式提交 2e58a9d 的综合分累计提高 0.00420612736。

### 分开估计市场与个股的预测修正

把三个期限的预测分别分解为 50 股均值和相对均值的个股部分，用前 16 天内的滚动预测估计它们与后来预测误差的关系。分别拟合市场对个股、个股对市场、个股之间的修正，按目标做滚动筛选。Y1 采用市场对个股的修正，Y2 采用三组修正，Y3 沿用前一阶段。完成上一节后，在 src 目录执行以下步骤。单日推断入口为 `released_model.SessionRefinement.predict`。

```bash
python -u experiment_joint_separated_peer_transfer_v1.py
python -u joint_separated_peer_references.py verify
python -u joint_separated_peer_references.py
```

前 16 天训练，后 4 天共 936,250 行全量验证和参数重载均得 0.3667699791952955。Y1、Y2、Y3 分别为 0.5010129407411067、0.38866985525067965、0.29993286894374055。相对前一节综合分提高 0.00015722028，第 17–20 天的变化分别为 +0.00040485、+0.00007784、−0.00005163、+0.00011555。相对上次正式提交 2e58a9d 的累计提高为 0.00436334764。

### 用 Y2 修正的近期变化更新 Y3

记上节模型对 Y2 的改变量为 z。用当前 z 和它相对最近 5 个时点均值的偏差调整 Y3，均值包含当前时点，并在午盘重置。系数由前 16 天内的滚动预测估计。按完整 Y3 与后 40 步对相关系数的贡献筛选。完成上一节后，在 src 目录依次运行以下命令，单日推断入口为 `released_model.SessionRefinement.predict`。

```bash
python -u experiment_y3_peer_carry_impulse_v1.py
python -u y3_peer_carry_impulse_references.py verify
python -u y3_peer_carry_impulse_references.py
```

前 16 天训练，后 4 天 936,250 行全量回放和参数重载得分均为 0.3667730852050137。Y1、Y2、Y3 的相关系数依次为 0.5010129407411067、0.38866985525067965、0.299939080963177。相对前一节提高 0.00000310601，两天提高、两天下降。按日期及 120 步时间块配对重采样的增量 95% 区间为 [−0.000171, +0.000181]。相对正式提交 2e58a9d 的累计提升为 0.00436645365。

### 估计盘口字段的共同变化，预测后 40 步

取 8 个带正负方向的字段组合、8 个反映字段变化幅度的量，以截至当前时点的 4、20、96 步历史构造输入。第一步预测未来 20 步各量的均值变化及 64 个两两乘积。对每一对，用预测的乘积减去两个预测均值的乘积，估计给定当前信息时的协方差。第二步用训练日随后第 21–60 步的收益估计各量与收益的关系。最后在训练期滚动预测上选择 Y3 的修正系数。每天推断只使用当时已经观察到的数据，午盘重新累计各时间窗口。

完成上一节后，在 src 目录运行以下命令。单日推断入口为 `released_model.SessionRefinement.predict`。

```bash
python -u experiment_y3_future_book_comoments_v1.py
python -u y3_future_book_comoment_references.py verify
python -u y3_future_book_comoment_references.py
python -u audit_y3_book_comoment_components_v1.py
```

前 16 天训练，后 4 天共 936,250 行全量回放和参数重载综合分均为 0.36692817088171936。Y1、Y2、Y3 的相关系数为 0.5010129407411067、0.38866985525067965、0.3002492523165882。相对前一节，综合分提高 0.00015508568。第 17–20 天分别变化 +0.00005137、+0.00046973、−0.00003369、+0.00007567。增量的配对日期与时间分块重采样 95% 区间为 [−0.0000365, +0.0004615]。相对上次正式提交 2e58a9d 的累计提升为 0.00452153933。

拆项对照中，只用均值时 Y3 为 0.29992650，只用协方差时为 0.30026286。发布候选沿用前 16 天滚动筛选选出的均值与协方差组合。

### 校准预测均值相乘产生的非线性项

上一阶段用两个预测均值的乘积估计独立变化时的共同变动。本阶段把这个乘积与使用相同输入和岭回归参数得到的线性近似作差。再用前 16 天的滚动预测，比较这个差值、字段剩余误差的共同变动以及均值预测误差三部分。滚动筛选选择了对均值乘积与其线性近似之间的差值单独调整 Y3。预测所需的每个量都在当前时点可计算。

完成上一节后，在 src 目录运行以下步骤。单日推断接口为 `released_model.SessionRefinement.predict`。

```bash
python -u experiment_y3_book_covariance_residuals_v1.py
python -u y3_book_covariance_residual_references.py verify
python -u y3_book_covariance_residual_references.py
```

前 16 天训练，后 4 天 936,250 行全量回放及参数重载综合分为 0.3670181075882376，Y1、Y2、Y3 分别为 0.5010129407411067、0.38866985525067965、0.30042912572962466。相对上一节综合分提高 0.00008993671。两节合计相对 0.3667730852050137 提高 0.00024502238，第 17–20 天分别变化 +0.00006446、+0.00077684、−0.00004918、+0.00010122。按日期和 120 步时间块重采样的两节合计增量 95% 区间为 [−0.0000851, +0.0007679]。相对正式提交 2e58a9d 累计提升 0.00461147604。

### 将 Y3 的盘口修正用于其他期限

分别用前 16 天的滚动预测检验三个期限。Y2 选择了以各训练日斜率中位数估计的系数，等于 1.60942576，将上一节产生的 Y3 修正量乘以该系数加到 Y2。Y1 和 Y3 采用上一节的输出。

在 src 目录依次运行。完整单日推断沿用 `released_model.SessionRefinement.predict`。

```bash
python -u experiment_joint_book_comoment_calibration_v1.py
python -u joint_book_comoment_references.py verify
python -u joint_book_comoment_references.py
```

前 16 天训练、后 4 天 936,250 行全量验证及参数重载综合分均为 0.36707508273355527。Y1、Y2、Y3 的相关系数分别为 0.5010129407411067、0.3888597724017387、0.30042912572962466。相对上一节提高 0.00005697515。本轮相对 0.3667730852050137 提高 0.00030199753，第 17–20 天的综合分增量分别为 +0.00006412、+0.00094528、−0.00008735、+0.00019316。按日期和 120 步块配对重采样，本轮增量的 95% 区间为 [−0.0001175, +0.0009467]。相对上次正式提交 2e58a9d 累计提高 0.00466845118。

### 分段预测盘口共同变化

将 Y3 的未来 60 步分成三个连续的 20 步收益段。在训练日期上，分别估计观察当前帧及再观察 10 步后，对未来盘口字段变化的预期。两个估计之差反映这 10 步带来的新增信息。用不同交易日拟合中间估计和后续读出，分别计算买卖压力与活跃度的预期变化如何共同影响三段收益。部署时，使用当前及此前的盘口预测这些共同变化。

训练期滚动筛选保留了第 21–40 和第 41–60 步的读出。再按历史预测的波动大小校准 Y2 和 Y3 的增量。冻结训练参数后，在第 17–20 天全部 936,250 行回放，综合分为 **0.3672143845689908**。Y1、Y2、Y3 的相关系数分别为 **0.5010129407411067、0.38892915060342914、0.30066610247948145**。四天已多次用于项目内版本比较。

完成上一节的历史预测重建后，在 `src` 目录运行以下命令。前两组命令分别训练并验证两个新增阶段。最后一条复核各部分对增益的贡献。单日推断入口为 `released_model.SessionRefinement.predict`。

```bash
python -u experiment_y3_interval_covariance_bridge_v1.py
python -u y3_interval_covariance_bridge_references.py verify
python -u y3_interval_covariance_bridge_references.py
python -u experiment_joint_interval_covariance_scale_v1.py
python -u joint_interval_covariance_scale_references.py verify
python -u joint_interval_covariance_scale_references.py
python -u audit_y3_interval_covariance_sources_v1.py
```

相对上一节的 0.36707508273355527，综合分提高 **0.00013930183544**，第 17 至 20 天分别提高 0.00003600、0.00025962、0.00017652、0.00010519。以日期和 120 步时间块配对重采样，本轮增量的 95% 区间为 [0.00001143, 0.00028375]。冻结参数的消融中，只使用按第 21–40 步收益训练的盘口项，综合分变化为 +0.00014960。三个收益段对 Y3 的实际相关系数贡献均提高。相对上次正式提交 `2e58a9d` 的 0.36240663155303376，综合分累计提高 0.00480775301596。

### 用其他股票的期限关系校准 Y3

把 50 只股票按固定顺序交替分为两组。每一帧先扣除三个预测各自的截面均值。用对侧 25 只股票分别估计 Y1、Y2 相对于 Y3 的回归斜率，再比较当前斜率与指数加权历史斜率。历史权重以 96 步为衰减时间常数，午后重新初始化。斜率差改变本组个股 Y3 的幅度。权重由前 16 天内按日期滚动生成的两组历史预测拟合。

先完成上一节的训练、验证及历史预测重建。在同一环境的 `src` 目录执行下列命令。第一条训练当前读出，自动回放第 17–20 天并重新加载权重。后续命令独立复算两组斜率，重建历史预测，并核对相对本地上一版和正式发布版的增益。单日推断入口为 `released_model.SessionRefinement.predict`。

```bash
python -u experiment_y3_dynamic_horizon_geometry_v1.py
python -u y3_dynamic_horizon_geometry_references.py verify
python -u y3_dynamic_horizon_geometry_references.py
python -u y3_dynamic_horizon_geometry_references.py audit
```

冻结模型在后四天共 936,250 行上的综合分为 **0.36726246698646897**。三个目标的相关系数为 **0.5010129407411067、0.38892915060342914、0.30076226731443784**。相对上一节的综合分提高 0.00004808241748，四天各自的变化为 +0.00002283、+0.00011393、+0.00031967、−0.00019234。配对重采样的增益 95% 区间为 [−0.00015901, +0.00027423]。相对正式提交 `2e58a9d`，综合分累计提高 0.00485583543344。

### 根据其他股票的预测状态估计 Y3 误差

对侧 25 只股票提供三个预测期限的波动幅度和相关性。第一层估计它们偏离近期水平的程度，并将接收股票的预测转换到近期的波动尺度和相关结构。第二层用历史日期的样本外预测，学习不同状态下三个预测与 Y3 误差的关系，同时在收益增量尺度上训练 Y1 的校正。第三层按短期和长期预测的方向是否一致给每只股票连续赋权，分别估计不同意见股票的状态，再结合接收股票自身的一致程度校准 Y3。

先完成上一节的训练和历史预测重建。在 `src` 目录运行以下命令。每个训练入口根据前 16 天的日期滚动结果选定配置，随后在后四天回放并重新加载参数。历史预测重建供下一层训练使用。单日预测入口为 `released_model.SessionRefinement.predict`。

```bash
python -u experiment_y3_horizon_covariance_transport_v1.py
python -u y3_horizon_transport_references.py verify
python -u y3_horizon_transport_references.py
python -u experiment_joint_sibling_horizon_skill_v1.py
python -u joint_sibling_horizon_skill_references.py verify
python -u joint_sibling_horizon_skill_references.py
python -u experiment_y3_horizon_agreement_skill_v1.py
python -u y3_horizon_agreement_skill_references.py verify
python -u y3_horizon_agreement_skill_references.py
python -u y3_horizon_agreement_skill_references.py audit
```

三层依次得到综合分 0.3673065117969436、0.3673981953960604、**0.3674923210527609**。最终在第 17–20 天全部 936,250 行上的相关系数为 **0.5010285545695456、0.38892915060342914、0.30121572991564616**。相对上一节提高 0.00022985406629，四天的综合分增量分别为 +0.00035147、+0.00058834、+0.00147148、−0.00141950。按日期和 120 步时间块配对重采样，增益的 95% 区间为 [−0.00106852, +0.00121265]。相对正式提交 `2e58a9d` 累计提高 0.00508568949973。

### 按盘口更新频率校准长期预测

每只股票根据可见的价格变动，比较最近 8 步与 128 步指数加权的更新频率，早盘、午后分别初始化。根据另一组 25 只股票的活跃度和三个期限预测的同向程度，估计不同盘口状态下当前预测与 Y3 误差的关系。接收股票自身的活跃度和方向一致程度决定额外调整幅度。所有误差系数均用前 16 天内按日期滚动产生的预测训练。

先完成上一节的历史预测重建，在 `src` 目录运行以下命令。第一条训练并全量回放，第二条用另一种计算方式核对选定读出和截断输入，第三条重建训练期滚动预测，最后两条分别统计总体增益和单项贡献。单日推断沿用 `released_model.SessionRefinement.predict`。

```bash
python -u experiment_y3_book_activity_skill_v1.py
python -u y3_book_activity_skill_references.py verify
python -u y3_book_activity_skill_references.py
python -u y3_book_activity_skill_references.py audit
python -u audit_y3_book_activity_skill_parts_v1.py
```

在第 17–20 天全部 936,250 行上重载参数回放，综合分为 **0.367542852386276**，三个目标的相关系数为 **0.5010285545695456、0.38892915060342914、0.3013167925826763**。相对上一节提高 0.00005053133352。四天综合分增量依次为 +0.00018678、+0.00021340、+0.00058628、−0.00076796，按日期和 120 步时间块配对重采样得到的增益 95% 区间为 [−0.00061088, +0.00048873]。在同一次冻结参数回放中，仅用预测分歧的对照读出使 Y3 相关系数提高 0.00009561，完整读出提高 0.00010106。相对正式提交 `2e58a9d` 综合分累计提高 0.00513622083324。

## 用 Y3 修正的时间尺度校准三个目标

前 16 天训练，使用其中第 9–16 天的两套滚动样本外预测训练最后一层。取四个模型阶段的 Y3 输出，依次计算三个相邻阶段的预测差。在每个交易时段内，用 6 步和 64 步指数均线提取预测差的快速变化、较慢变化和持续成分。Y2 选用慢速持续成分，Y1 和 Y3 还利用短期预测与长期预测方向的一致程度。候选和参数均通过前 16 天的滚动切分确定。推断沿用 `released_model.SessionRefinement.predict`，在一次前向计算中保留四处所需输出。

先完成前面各节。进入 `src` 后按以下顺序训练新阶段、重载参数并核对实现、重建训练期滚动预测和统计增益。

```bash
python -u experiment_joint_revision_timescale_transfer_v1.py
python -u joint_revision_timescale_references.py verify
python -u joint_revision_timescale_references.py
python -u joint_revision_timescale_references.py audit
```

第 17–20 天全量回放共 936,250 行。Y1、Y2、Y3 的 Pearson 相关系数依次为 0.5010764669129306、0.3890565599823692、0.3013352161628069，按 20%、30%、50% 加权得 **0.3675998694587003**。相对上一本地版本 0.367542852386276，增益为 0.0000570170724243。第 17–20 天的单日分数依次为 0.409854、0.323161、0.358563、0.368967，其中第 20 天下滑约 0.00199。这四天已反复用于版本比较，后续日期的泛化仍待新数据验证。

若要核对各部分对相关系数变化的贡献，在 `src` 中运行 `python -u audit_revision_timescale_contributions.py`。它固定已训练系数，逐一计算所有特征组组合的分数，用 Shapley 值分摊增益。Y2 的 +0.00012741 全部来自慢速持续成分。Y3 的短长期一致程度项贡献约 +0.0000375，慢速持续成分贡献约 −0.0000253。它们描述这四天内的分数变化。

## 跨市场价格变动方向与 Y3 远段预测

本版用前 16 天拟合，第 17–20 天全量回放 936,250 行。综合分为 0.3676547047120568，距离 0.38 还有 0.01234530。上一轮本地最佳为 0.3675998694587003，上次远端发布为 0.36240663155303376。Y3 从 0.3013352162 升至 0.3014448867，Y1 和 Y2 的输出保持一致。四天相对上一轮的综合分变化依次为 +0.00015729、+0.00003804、+0.00002302、−0.00003321。相邻 120 步分块重采样给出的增益区间跨过零，最新的小幅提升仍需新交易日检验。

对每只股票，记录最近四次价格方向是否延续、近期方向偏向、活动分组及距上次变动的步数。用本赛题训练日估计变动的到达概率和幅度，把 FI-2010 的方向统计与本地统计融合，递推未来 60 步的期望变动。训练日滚动筛选选中用两套估计的第 21–60 步差值修正 Y3；四天合并时，远段相关系数从 0.14164962 升至 0.14183620。筛选时也计算 LOBSTER 美国股票对照。训练需要按前文准备 FI-2010 压缩包和五只股票的 LOBSTER 样例压缩包。推理所需的方向表、活动分组、修正系数由训练入口写入本地参数文件。

完成上一节的训练与历史预测重建后，在本目录的 `src` 下依次运行：

```bash
python -u experiment_y3_cross_market_marked_renewal_v1.py
python -u y3_cross_market_renewal_references.py verify
python -u y3_cross_market_renewal_references.py
python -u y3_cross_market_renewal_references.py audit
```

训练、全量回放、参数重载和分段独立复算的结果写入本地 `artifacts/project_restart_generalization/y3_cross_market_marked_renewal_v1/`。日内预测沿用 `released_model.SessionRefinement.predict`，最终阶段从训练文件载入参数。

## 分别估计首次变价和后续变价对三个时间尺度的影响

使用前 16 个交易日训练，在第 17–20 天的 936,250 行上合并计算相关系数。Y1 为 0.50107647，Y2 为 0.38921074，Y3 为 0.30157157，综合分为 0.3677642988431156。上一版本地得分为 0.3676547047120568，累计较上次远端发布的 0.36240663155303376 提升 0.00535766729008184。四天相对上一本地版本的综合分变化依次为 +0.00030008、+0.00011268、−0.00003065、−0.00002821。验证日参与过多轮模型比较，仍需新交易日检验泛化表现。

模型在当前股票的成交活跃度分组、最近价格变化方向、距上次变动步数上查表。国内数据给出等待时间和变动幅度，FI-2010 提供方向统计。动态规划分别对未来 5、20、60 步累计定价，并计算只让首次变价、只让后续变价、或让所有变价采用跨市场方向统计时的期望差异。第 9–12 天和第 13–16 天的连续四日滚动检验分别对三个目标选型。Y2 采用首次变价后才引入外部方向信息的 20 步估计，Y3 采用覆盖所有变价的 60 步估计，Y1 沿用上版。修正按当前 50 股截面去均值。

实际加入原预测的是两套转移规则得到的期望收益之差。Y2 使用“首次变价后改用跨市场方向估计”减去“全程采用本地方向估计”，拟合系数约为 −0.05064。Y3 使用“全程跨市场方向估计”减去“全程本地方向估计”，系数约为 −0.01396。因此外部与本地估计分歧越大，模型沿该差值的反方向修正原预测。

按前文重建上一版的训练产物和滚动预测，再在本目录的 `src` 依次运行以下命令。第一条使用仅前四天训练的 BookNet 和事件 GRU 生成第 5–8 天的早期预测；相应权重缺失时会在本地训练。需要 CUDA，执行环境和竞赛数据路径见文首。

```bash
python -u y3_four_day_neural_references.py
python -u experiment_joint_first_event_transfer_v1.py
python -u joint_first_event_references.py verify
python -u joint_first_event_references.py
python -u joint_first_event_references.py audit
```

`released_model.SessionRefinement.predict` 是组合模型的推理入口。训练选择、全量回放、参数重载与逐时点独立复算的结果保存在本地 `artifacts/project_restart_generalization/joint_first_event_transfer_v1/`。

## 当前版本的五步预测

在上述模型上，第一个目标按当前盘口更新速度修正未来五步内首次变价的幅度。速度分组使用价格和盘口变化的快慢指数均值，等待时间和上涨、下跌幅度均由前 16 天估计。逐时点复核独立计算“首次变价出现在第几步”的概率，并分别乘以当时对应的预期幅度修正。

按上一节生成全部参数与历史预测后，在 `src` 下执行：

```bash
python -u experiment_joint_conditional_first_wait_v1.py
python -u joint_conditional_wait_references.py verify
python -u joint_conditional_wait_references.py
python -u joint_conditional_wait_references.py audit
```

组合推理入口为 `released_model.SessionRefinement.predict`。前 16 天训练、第 17–20 天共 936,250 行合并验证的 Y1、Y2、Y3 相关系数分别为 0.5010773355307029、0.3892107411631115、0.3015715662232022，综合分为 0.3677644725666751。相比上一节只提升约 0.00000017。相比远端版本 0.36240663155303376，累计提升 0.00535784101364134；这一版的主要增益来自前面各阶段。完整回放及重载结果在本地 `artifacts/project_restart_generalization/joint_conditional_first_wait_v1/`。


## 通过五只股票的同步逐笔数据估计成交方向

这一版给 Y3 增加一个校正项。外部训练使用 LOBSTER 的 AAPL、AMZN、GOOG、INTC、MSFT 在 2012-06-21 的 Level 1 消息与盘口快照，按同一组五秒端点对齐。用主动成交和撤单消息分别构造方向监督。先估计只观察个股的模型，再估计同时观察四只同组股票的模型。在国内数据上用三种固定的五股分组取平均。最终使用未来第 21–60 个外部时间格的成交方向估计差值，与预测时可见的本地盘口不平衡度组合，去掉同帧 50 股均值，再加到 Y3 上。国内字段已匿名，盘口不平衡度用字段 21 在前四天取值范围内线性缩放至 [−1, 1] 近似。设模型差值为 f、当前盘口不平衡度代理为 b，所选信号为 `f × tanh(2 f b)`，翻转 f 的符号给出完全相同的结果。因而去截面均值之前，方向由盘口不平衡度给出，外部模型贡献的是幅度。外部时间格与本题时间步分别按各自的数据定义计算。

国内第 1–16 天用于训练和校准。第 9–12、13–16 天构成两个先训练后验证的内部窗口。最终冻结参数，在第 17–20 天的全部 936,250 行合并得到三个相关系数 0.5010773355307279、0.3892107411630908、0.30157382121505955，加权分 **0.36776560006260256**。相对上一节约增加 0.00000113，相对上次远端正式版本 0.36240663155303376 累计增加约 0.005359。四个验证日里三天微升、一天微降。按日及 120 步时间块重采样，本次增益的 2.5% 至 97.5% 分位数约为 −0.00000224 至 +0.00000553。这四个验证日已经用于历次模型比较。

准备上一节的历史预测及参数，并将五个 Level 1 压缩包放入本机的 `src/artifacts/project_restart_generalization/external_event_prior_v1/lobster_samples/`，依股票代码命名，例如 `AAPL.zip`。每个压缩包保留相同日期的原始 message 和 orderbook CSV。训练脚本记录各文件 SHA-256 及外部最后三分之一时段的检验结果。样本来源见 [LOBSTER 数据样例](https://lobsterdata.com/info/DataSamples.php)。

在 `src` 下运行以下命令。第一条包含末级参数训练、后四天全量回放和重载验证。后续命令分别独立计算所选成交信号、重建滚动参考预测及估计按交易日和时间块重采样的增益区间。

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python -u experiment_y3_external_panel_message_origins_v1.py
python -u y3_panel_message_origin_references.py verify
python -u y3_panel_message_origin_references.py
python -u y3_panel_message_origin_references.py audit
python -u audit_y3_external_flow_magnitude_v1.py
```

训练期的幅度对照固定盘口方向，将强度替换为另一只股票同期的强度。两个滚动窗口、两组历史参考预测上，实际强度对 Y3 的增益均为正，幅度很小。

完整组合推理使用 `released_model.SessionRefinement.predict`。末级结果在 `artifacts/project_restart_generalization/y3_external_panel_message_origins_v1/`。Git 版本只包含训练源码与使用说明。

## 同期相似盘口的成交强度对照

每个时点把交易方向代理固定为待预测股票的盘口不平衡度，分别代入自己的估计成交强度和其他 49 只股票的强度。参照值给盘口不平衡度更接近的股票更高权重，并加入相当于两只股票的全体同伴均值以减小波动。使用自己的信号与参照值之差，在同帧去均值后校正 Y3。计算只用当前及历史观测值。强度、盘口字段及外部样本的含义见上一节。

在第 1–16 天训练。两次滚动比较的验证窗口分别为第 9–12 天和第 13–16 天。最终冻结参数，在第 17–20 天全部 936,250 行得到 Y1 **0.5010773355307279**、Y2 **0.3892107411630908**、Y3 **0.301575532673198**，加权综合分 **0.3677664557916718**。本阶段提高约 0.00000086，两天上升、两天下降。按交易日及 120 步时间块重采样，增益的 2.5% 至 97.5% 分位数为 −0.00000733 至 +0.00000952。相对上次远端正式版本 0.36240663155303376，累计提高约 0.005360。后四天已用于历次模型对比。

在上一节生成参数与历史参考预测后，从 `src` 目录运行以下命令。训练入口拟合末级系数，执行全量回放与重载。复核脚本可逐个参照股票独立计算校正，重建内部窗口的预测并估算区间。组合推理入口为 `released_model.SessionRefinement.predict`。

```bash
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
python -u experiment_y3_conditional_flow_exchange_v1.py
python -u y3_flow_exchange_references.py verify
python -u y3_flow_exchange_references.py
python -u y3_flow_exchange_references.py audit
```

本阶段本机结果目录为 `artifacts/project_restart_generalization/y3_conditional_flow_exchange_v1/`。


## 约束短期贡献的 Y3 价格轨迹解码器

输入每帧 50 只股票的当前数据和过去 180 步内的盘口统计。四个解码器以五步为间隔预测未来 12 个价格位置，用逐点误差和频域误差训练。它们分别输出市场和个股的轨迹。末级只调整 Y3，使用这些轨迹和已有模型的三个预测。训练目标是增加第 21–60 步对 Y3 相关系数的贡献，同时约束前 20 步在拟合窗口内的贡献。约束在新日期的兑现程度通过回放单独检查。

使用前 16 天训练，第 9–12、13–16 天为训练期内的滚动对照。冻结全量模型后，第 17–20 天全部 936,250 条样本合并计算得到 Y1 **0.5010773355307279**、Y2 **0.3892107411630908**、Y3 **0.3016499968532272**，加权分 **0.3678036878816864**。相对前一本地最优提高 **0.000037232090014605035**，相对 GitHub 提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的 **0.36240663155303376** 累计提高约 **0.00539706**。第 17、18 天的单日分下降，第 19、20 天上升。按交易日及 120 步时间块进行配对重采样，本次增量的 2.5% 和 97.5% 分位数为 −0.000294、+0.000421。这四天已用于多轮模型比较。

先完成上一节的训练及历史参考预测生成。在本节和上文的 `src` 目录中运行以下命令，需要 CUDA。第一条从头训练本版解码器、拟合修正系数并执行冻结参数回放，第二条单独从磁盘加载参数重放。第三条检查分段收益贡献和相对正式版本的增益。第四条按训练窗口重建后续实验需要的历史预测。

```bash
python -u experiment_y3_protected_path_decoder_v1.py
python -u experiment_y3_protected_path_decoder_v1.py replay
python -u y3_protected_path_references.py
python -u y3_protected_path_references.py history
```

完整组合使用 `released_model.SessionRefinement.predict`。末级的权重、回放记录和历史预测在本地的 `artifacts/project_restart_generalization/y3_protected_path_decoder_v1/` 生成。[末级训练](src/experiment_y3_protected_path_decoder_v1.py) · [解码器](src/experiment_y3_frequency_path_loss_v1.py) · [回放检查](src/y3_protected_path_references.py)


## 用收益分组概率修正 Y3

在每个训练时点，把 50 只股票未来三个区间的收益分别由低到高分为十组，每组五只。收益相同的股票在跨组时平分概率。模型读入当前及过去 180 步以内的盘口汇总，估计每只股票属于各组的概率，同时估计十组的平均收益。训练期比较直接回归、独立概率和按截面组容量调整的概率。最后使用两种概率预测的差异修正第 21–60 步的个股预测，修正幅度在历史滚动预测上拟合。推理只需当前帧和先前的观测。

使用前 16 天训练，在第 9–12、13–16 天的滚动对照上选择修正方式。全部参数冻结后，用第 17–20 天合并的 936,250 行计算，Y1 **0.5010773355307279**、Y2 **0.3892107411630908**、Y3 **0.30168571788242204**，综合分 **0.36782154839628384**。相比前一本地版 **0.3678036878816864** 提高 **0.000017860514597428656**。第 17、18 天分别下降约 0.00060 和 0.00089，第 19、20 天分别上升约 0.00079 和 0.00116。相对上一次正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的 **0.36240663155303376**，四天综合分累计提高约 **0.00541492**。这四天已用于多轮模型比较。

完成上一节各阶段的训练、回放和历史预测生成后，在同一 `src` 目录使用 CUDA 运行下列命令。第一条重新训练分组模型并回放，第二条单独加载参数验证，第三条用两种等价方式计算分组概率修正并核对逐日得分。最后一条生成后续实验可用的历史滚动预测。

```bash
python -u experiment_y3_joint_rank_payout_v1.py
python -u experiment_y3_joint_rank_payout_v1.py replay
python -u y3_rank_payout_references.py
python -u y3_rank_payout_references.py history
```

本地结果目录为 `artifacts/project_restart_generalization/y3_joint_rank_payout_v1/`。整套模型由 `released_model.SessionRefinement.predict` 调用。[训练与推理代码](src/experiment_y3_joint_rank_payout_v1.py) · [回放与独立核验](src/y3_rank_payout_references.py)


## 在推理时校准收益分组的概率

直接从独立概率模型得到每只股票的十组概率，再交替调整每组的总概率和每只股票的概率和，共四次。每组的总概率目标为五，每只股票的概率和为一。把调整前后的概率差乘以两个模型对该组收益的平均估计，再对十组求和，得到第 21–60 步的修正量。这一版从价格轨迹解码器的 Y3 输出开始计算，沿用上一节在前 16 天拟合的网络参数和缩放系数。

训练窗口为第 1–16 天。第 9–12、13–16 天的滚动对照用于检验概率校准。第 17–20 天合并验证 **936,250 行**，Y1 **0.5010773355307279**、Y2 **0.3892107411630908**、Y3 **0.3023172856352974**，综合分 **0.3681373322727215**。相对完整概率修正的 **0.36782154839628384** 提高 **0.0003157838764376586**，相对价格轨迹解码器的 **0.3678036878816864** 提高 **0.000333644391035115**。相对 GitHub 正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的累计提升约 **0.00573070**。该版本源自已查看后四天结果的消融实验，这组日期已多次用于比较。

在上一节的模型权重以及价格轨迹解码器的历史参考预测准备好后，在 `src` 目录运行以下命令。第一条在训练期重建概率对照、固定读出方式并执行全量回放。第二条重新加载参数，第三条用独立实现核对结果并比较正式版。第四条重建后续实验所需的训练期预测。

```bash
python -u experiment_y3_rank_capacity_readout_v1.py
python -u experiment_y3_rank_capacity_readout_v1.py replay
python -u y3_rank_capacity_references.py
python -u y3_rank_capacity_references.py history
```

完整推理入口为 `released_model.SessionRefinement.predict`。本地回放记录位于 `artifacts/project_restart_generalization/y3_rank_capacity_readout_v1/`。[概率校准与推理](src/experiment_y3_rank_capacity_readout_v1.py) · [训练期对照](src/audit_y3_rank_capacity_effect_v1.py) · [复核入口](src/y3_rank_capacity_references.py)


## 用已结算区间修正个股分组概率

在每个交易时段内，每隔 40 步更新一次修正。对每只股票、每个收益组，分别计算最近 360 步、1,440 步中实际归组次数和模型预期次数的比值。用两者的对数差调整当前概率，再恢复每组的截面容量。更新只使用已经走完 60 步的历史起点，每隔四步取一个起点；各组计数含平滑先验。实际归组由已观测的价格路径估计。60 步尚未走完时，算法继续使用此前已经结算的区间。

前 16 天训练，第 9–12、13–16 天做历史滚动比较。所选修正在两个窗口、两套参考预测上合并计算的 Y3 增量均为正，单日结果有波动。冻结训练参数后，在第 17–20 天全部 936,250 行上得到 Y1 **0.5010773355307279**、Y2 **0.3892107411630908**、Y3 **0.30245293685589475**，综合分 **0.36820515788302016**。相比前一本地最优 **0.3681373322727215** 提高 **0.0000678256102986663**，第 17、18、19 天逐日得分上升，第 20 天下降。相对 GitHub 正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的累计提升约 **0.00579853**。后四天已用于多轮模型比较。

先完成上一节的训练、回放和历史预测生成。进入 `src` 后运行以下命令。第一条训练个股校准参数并执行冻结回放。第二条从磁盘重新加载参数验证。第三条按每个预测时点独立重算允许使用的历史区间，并比较正式版得分。第四条重建后续实验所需的训练期历史预测。

```bash
python -u experiment_y3_settled_rank_probabilities_v1.py
python -u experiment_y3_settled_rank_probabilities_v1.py replay
python -u y3_settled_rank_probability_references.py
python -u y3_settled_rank_probability_references.py history
```

完整推理入口为 `released_model.SessionRefinement.predict`。本地结果保存在 `artifacts/project_restart_generalization/y3_settled_rank_probabilities_v1/`。[训练与推理](src/experiment_y3_settled_rank_probabilities_v1.py) · [独立回算与复核](src/y3_settled_rank_probability_references.py)

## 利用近期预测方向的变化细调 Y3

每次先去除 50 股 Y3 预测的截面均值和沿当前 Y2 方向的分量。取本交易时段内相隔 2、8、24、64 步的历史预测，估计截面方向的变化速度，以及连续两段速度之差。用训练期历史滚动预测拟合修正系数，把修正后的预测恢复到原有幅度。每帧的 Y3 均值、标准差及与 Y2 的截面协方差保持一致。

前 16 天训练，第 9–12 天与第 13–16 天作滚动筛选。冻结参数后，第 17–20 天共 936,250 行的相关系数为 Y1 **0.5010773355307279**、Y2 **0.3892107411630908**、Y3 **0.30245428146703784**。加权综合分 **0.3682058301885917**，相比前一本地版本 **0.36820515788302016** 高约 **0.0000006723**。第 17、18 天上升，第 19、20 天下降。相对正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的累计提升约 **0.00579920**。后四天已用于多轮模型比较。

先完成上一节的回放和历史预测生成。在 `src` 目录中依次运行训练、参数重载验证、独立逐时点复算和历史预测生成。

```bash
python -u experiment_y3_angular_forecast_history_v1.py
python -u experiment_y3_angular_forecast_history_v1.py replay
python -u y3_angular_forecast_references.py
python -u y3_angular_forecast_references.py history
```

完整推理入口为 `released_model.SessionRefinement.predict`。本地运行结果写入 `artifacts/project_restart_generalization/y3_angular_forecast_history_v1/`。[训练与推理](src/experiment_y3_angular_forecast_history_v1.py) · [独立复核与历史预测](src/y3_angular_forecast_references.py)

## 由同期其他股票的预测估计分组收益

每个时刻按四项可观测的盘口与近期交易特征分别将 50 股分为五组。模型学习预测每只股票在 20 步后的分组概率。推理时将 50 股分成两个互斥的 25 股集合，用一半股票的当期 Y3 预测估计从各起始组到各目的组的相对收益，再给另一半股票计算组内修正。交换两组角色，在两种预设划分下取均值，并对截面中心化。前 16 天滚动预测拟合的系数为 **−0.9720812113604536**，因此会从原预测中扣除这项组内信号。后四天修正量的截面均方根约为原 Y3 个股相对市场预测均方根的 0.585%。训练使用历史日期的后续分组作为监督，推理使用当时可见的盘口和预测值。

冻结参数后，第 17–20 天共 936,250 行的相关系数分别为 Y1 **0.5010773355307279**、Y2 **0.3892107411630908**、Y3 **0.30249134943732225**，加权综合分为 **0.36822436417373394**。前一本地版本为 **0.3682058301885917**。第 18、20 天逐日得分上升，第 17、19 天下降。与 GitHub 正式版本 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 相比累计提高约 **0.00581773**。后四天已用于多轮模型比较。

在上一节的训练、校验和历史预测生成结束后，进入 `src`，依次运行以下命令以训练、重载回放、独立复核并重建第 9–16 天的滚动预测。

```bash
python -u experiment_y3_donor_migration_prices_v1.py
python -u experiment_y3_donor_migration_prices_v1.py replay
python -u y3_donor_migration_references.py
python -u y3_donor_migration_references.py history
```

完整推理入口为 `released_model.SessionRefinement.predict`。本地结果写入 `artifacts/project_restart_generalization/y3_donor_migration_prices_v1/`。[训练与推理](src/experiment_y3_donor_migration_prices_v1.py) · [分组预测网络](src/experiment_y3_book_group_migration_v1.py) · [独立复核](src/y3_donor_migration_references.py)

## 用日内跟踪组合修正个股 Y3

每只股票用其余 49 只股票构造一个非负、权重和为 1 的组合。前四天用于估计固定权重。交易时段内每 60 步更新一次日内权重。若更新点为 t，拟合样本最迟到 t−145，随后用 t−144 到 t−49 的独立历史区间比较日内组合和固定组合对这只股票的跟踪误差。用 8 步和 24 步两种回报跨度衡量误差，只在平均跟踪误差改善时启用修正。每个交易时段的前 360 步使用原预测。

以两种组合对近期价格变化、当期 Y3 和远期分量预测的差异作为输入，用前 16 天内的滚动预测拟合 Y3 修正。冻结后在第 17–20 天共 936,250 行得到 Y1 **0.5010773355307279**、Y2 **0.3892107411630908**、Y3 **0.3024974013241494**，综合分 **0.3682273901171475**。前一本地版本为 **0.36822436417373394**。第 18 天上升，第 17、19、20 天下降。相对 GitHub 正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 累计提升约 **0.00582076**。后四天已用于多轮模型比较。

先运行上一节的训练、回放和历史预测生成命令。进入 `src` 后依次运行以下命令，以训练模型、重载验证、独立复算和生成滚动预测。

```bash
python -u experiment_y3_intraday_replication_v1.py
python -u experiment_y3_intraday_replication_v1.py replay
python -u y3_intraday_replication_references.py
python -u y3_intraday_replication_references.py history
```

完整推理入口为 `released_model.SessionRefinement.predict`。本地回放报告保存在 `artifacts/project_restart_generalization/y3_intraday_replication_v1/`。[模型训练与推理](src/experiment_y3_intraday_replication_v1.py) · [独立复算与历史预测](src/y3_intraday_replication_references.py)

## 用上午已经兑现的收益修正下午的个股预测

将上午分为 180 步区间，在每个区间的前 120 步抽取预测起点。用截至午间已经观测的同期股票价格，估计每只股票在整个 Y3 窗口及第 21–60 步的平均预测误差。两项误差方向一致时，取绝对值较小的那项，扣除同期 50 股均值，乘以训练期拟合的系数，从下午开盘起加入 Y3 预测。收益使用已观测单步回报复算，缺失区间从估计中剔除。系数及规则在前 16 天内按时间顺序筛选。

第 17–20 天的 936,250 行全量回放和参数重载均得到 **0.3682278818078105**，三个相关系数为 **0.5010773355307279、0.3892107411630908、0.30249838470547546**。上一版本综合分为 **0.3682273901171475**，第 17、18 天上升，第 19、20 天下降。按交易日和 120 步时间块进行配对重采样，增量的 95% 区间约为 −0.00001117 至 +0.00001091。后四天已用于多轮模型比较。与 GitHub 正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 相比累计提高约 **0.00582125**。

完成上一节的训练、复核和历史预测生成后，在 `src` 目录运行以下命令。第三条命令逐项复算午间修正，并比较只读到下午第一个时点及下午盘中时点的结果。最后一条生成后续训练所需的滚动预测。完整推理入口为 `released_model.SessionRefinement.predict`。

```bash
python -u experiment_y3_morning_bias_transfer_v1.py
python -u experiment_y3_morning_bias_transfer_v1.py replay
python -u y3_morning_bias_references.py
python -u y3_morning_bias_references.py history
```

回放、逐日得分和训练期筛选结果写入 `artifacts/project_restart_generalization/y3_morning_bias_transfer_v1/`。[训练与推理](src/experiment_y3_morning_bias_transfer_v1.py) · [独立复核](src/y3_morning_bias_references.py)

## 按盘口状态调整 Y1 预测幅度

模型利用 16 个核心字段及其最近 6、80 步均值解释其余 84 个字段，再从剩余信号中学习与未来第 21–60 步盘口状态相关的 10 个方向。推理时根据每只股票当前在这些方向上的绝对值相对 50 股均值的差异，调整 Y1 预测偏离同期市场均值的幅度。历史均值在每个交易时段起点重置。目标与调整方式在前 16 天的两个滚动窗口内选择。

第 17–20 天全量回放及参数重载的综合分均为 **0.3682318666137022**，对应 Y1、Y2、Y3 相关系数分别为 **0.5010972595601513、0.3892107411631115、0.302498384705477**。相对上一节增加 0.0000039848058917，Y2 和 Y3 的预测沿用上一节模型。与正式版本 2e58a9d 相比累计提升约 0.00582524。

完成上一节命令后，在 `src` 目录训练最后一层并重载复核。完整推理入口为 `released_model.SessionRefinement.predict`，输出报告位于 `artifacts/project_restart_generalization/joint_field_state_amplitudes_v1/`。

```bash
python -u experiment_joint_field_state_amplitudes_v1.py
python -u experiment_joint_field_state_amplitudes_v1.py replay
```

## 用已兑现的股票对收益校准 Y3

在前 16 天内部，用第 9–12 天和第 13–16 天两段滚动预测筛选修正规则。模型在同一时刻按已有 Y3 预测给 50 只股票排序，比较相隔 1、3、5、8、12、17 个名次的股票对。每过 128 步，使用已经完整走过 60 步的价格记录，估计不同 Y1、Y2、Y3 预测差距下的实际收益差。在股票编号奇偶划分的两组之间交叉使用这些估计，再把股票对修正换算成每只股票的 Y3 调整。每个交易时段独立累计历史。模型降低近似并列的股票对的权重，并在分组边界内插值，使微小的输入变化只引起较小的输出变化。

按前 16 天训练、后 4 天全量验证，在 936,250 条记录上，原始数据回放与重新加载参数的综合分均为 **0.3682524381591154**。Y1、Y2、Y3 的相关系数分别为 **0.5010972595601513、0.3892107411631115、0.3025395277963034**。相对上一节的综合分提高 **0.0000205715454132**，第 17 至 20 天的综合分分别提高约 0.00000515、0.00004619、0.00000711、0.00002667。该验证集已用于多轮模型比较。

完成上一节的模型训练与回放后，在本版 `src` 目录运行。第一条命令筛选修正规则、保存本地参数并回放，第二条独立重载。完整推理入口为 `released_model.SessionRefinement.predict`。结果保存在本机的 `artifacts/project_restart_generalization/y3_settled_rank_graph_v1/`。

```bash
python -u experiment_y3_settled_rank_graph_v1.py
python -u experiment_y3_settled_rank_graph_v1.py replay
```

[训练与回放](src/experiment_y3_settled_rank_graph_v1.py) · [股票对与名次的对应关系](src/y3_rank_spread_geometry.py)

## 用持续的市场预测校准个股 Y3

每个时点，对 50 股预测取均值，得到整体 Y3 和远段收益两条市场预测。远段收益从 Y3 中扣除按收益尺度折算的 Y2。沿用训练期既定的组合比例，开盘骨干模型占 65%，三个神经网络的均值占 35%。每个交易时段内只使用截至当前的最近 96 步，按 58% 和 42% 合并两种期限，依据前 16 天估计的个股市场敏感度生成修正，再扣除修正的截面均值。该层只调整 Y3。训练期间每两天重新拟合一次，并用打乱个股权重对应关系的对照检查收益是否依赖个股身份。

前 16 天训练、第 17–20 天全量验证，共 936,250 行。全量回放与参数重载的结果逐日一致。综合分从 **0.3682524381591154** 升至 **0.36828964169588124**，Y3 相关系数从 **0.3025395277963034** 升至 **0.3026139348698351**。四天中三天提高，第 18 天综合分下降约 0.00020103。对交易日及日内 120 步区块进行 2,500 次配对重采样，综合分增量的 95% 区间约为 −0.000184 至 +0.000271。后四天已用于多轮模型选择，尚需新的交易日验证泛化性。

完成上一节的训练，并生成前文所述的开盘模型滚动预测和神经网络预测缓存，在本版 `src` 目录执行以下命令。回放报告保存在 `artifacts/project_restart_generalization/y3_persistent_backbone_exposure_v1/`。完整推理入口为 `released_model.SessionRefinement.predict`，复用同一次推理已经计算的四组上游预测。

```bash
python -u y3_settled_spread_references.py
python -u experiment_y3_persistent_backbone_exposure_v1.py
python -u experiment_y3_persistent_backbone_exposure_v1.py replay
python -u audit_y3_persistent_exposure_transfer_v1.py
```

[训练与推理](src/experiment_y3_persistent_backbone_exposure_v1.py) · [个股敏感度估计](src/experiment_y3_instrumented_market_loadings_v1.py) · [逐日复核及配对重采样](src/audit_y3_persistent_exposure_transfer_v1.py)

## 用相同时段的历史盘口修正市场共同收益

第 17–20 天的 936,250 行合并计算后，综合分为 **0.3684284194485103**，Y3 相关系数为 **0.3028914903751069**。相对上一节的本地版本，综合分增加 **0.00013877775262902126**。相对 GitHub 上一正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的 0.36240663155303376，累计提高 **0.00602178789547653**。后四天已多次用于模型比较。

新层把每个交易时段按 192 步分段。对每段起点，汇集历史交易日从该位置开始最多 1,152 步的样本，在每个时段末尾预留完整的 60 步标签窗口。输入取当前三项预测的市场均值、最近 48 步的预测均值，以及过去 6 步和 32 步的价格变化与四对盘口字段差值。岭回归分别预测市场整体的 Y3 误差和第 21–60 步的误差。推理按当前时段取对应系数，以 62% 和 38% 合并两项修正，再乘以 0.23，加到 50 只股票的 Y3 上。所有滚动输入均在时段起点重置。

前 16 天用于训练。最后一层从第 5 天开始收集训练期滚动预测；在第 9–12 天、第 13–16 天的严格按日期截断的对照中，Y3 相关系数分别提高 0.006212 和 0.001036。另将系数冻结在第 8 天，两个窗口分别提高 0.005762 和 0.000878。最终在第 17 天前重新拟合系数。使用独立副本重训最后一层，系数数组逐项相同。

| 验证交易日 | 综合分 | 相对上一节的变化 |
| --- | ---: | ---: |
| 17 | 0.408737 | −0.001313 |
| 18 | 0.325073 | +0.002273 |
| 19 | 0.359566 | −0.000508 |
| 20 | 0.369997 | −0.000749 |

以 Y3 减去按收益尺度换算的 Y2 表示第 21–60 步收益，这一远段的全样本相关系数从 **0.142890** 升至 **0.143877**，其中 50 股市场均值的相关系数从 **0.215469** 升至 **0.218988**。可运行 `python -u audit_y3_session_market_band_attribution_v1.py` 复核。此项只用于误差归因，比赛得分使用完整 Y3。

去掉第 18 天，其余三天合并后的综合分变化为 −0.000907。按交易日和日内 120 步区块做 2,500 次配对重采样，四天合并的增量 95% 区间约为 −0.001478 至 +0.002095。

完成上一节的训练及回放后，在本版 `src` 目录依次运行以下命令。第一条重建前 16 天历史验证并拟合参数，第二条以缓存的上游预测对照，第三条回放完整四天并重载参数复算，第四条核对增量的不确定性。历史阶段的回放会读取已按原始输入摘要核验的上游结果。报告及本地参数写入 `artifacts/project_restart_generalization/y3_session_continuation_ridge_v1/`。完整组合的调用入口是 `released_model.SessionRefinement.predict(raw, base, opening)`，沿用上游的基础预测与开盘模型预测。单独应用本层可用 `correction()(raw, p)` 得到形状为（时点数，50）的 Y3 增量，此处 `p` 为上一节的完整三目标预测。

```bash
python -u experiment_y3_session_continuation_ridge_v1.py train
python -u experiment_y3_session_continuation_ridge_v1.py screen
python -u experiment_y3_session_continuation_ridge_v1.py verify
python -u audit_y3_session_market_uncertainty_v1.py
```

[本层的训练和预测](src/experiment_y3_session_continuation_ridge_v1.py) · [四天回放与重载](src/y3_marginal_replay_v1.py) · [按日和区块的增量复核](src/audit_y3_session_market_uncertainty_v1.py)

独立从原始 parquet 重建时，准备好本机已训练的基础包 `causal_16d` 及其 `weights/`，将下方路径换成本机位置，在 `src` 目录运行。脚本先导出可见的 100 个字段，重算四天及各自截断前缀的基础预测和开盘预测，再调用完整组合。最后读取标签评分，检查前缀误差，并与分阶段回放结果逐日比较。[完整推理复核入口](src/replay_session_market_release_v1.py)

```bash
python -u replay_session_market_release_v1.py --base-src /path/to/causal_16d/src --data project/data/public.parquet
```

使用独立源码副本从原始行情回放 936,250 行，综合分为 **0.36842841943968935**。四天截断前缀的最大预测差为 8.34e-07，逐日综合分与分阶段回放的最大差为 3.96e-11。

## 用股票组合预测 Y3 的个股差异

前 16 天训练，在第 9–12 天和第 13–16 天做先后两个滚动对照，最终在第 17–20 天的全部 936,250 行上计算比赛指标。新模型增加两步。

1. 每个时点按五个可观测特征给 50 只股票赋予正负权重。五个特征分别是近期与较慢价格变化之差、盘口两侧压力之差、慢速价格变化、绝对价格变化和 Y2 与 Y3 的预测分歧。每组权重减去截面均值，并去除与当前 Y3 预测截面的线性重叠。使用相近交易时段的历史样本，学习当前市场状态和组合短期预测与后续组合收益误差的关系。最后按每只股票的五项权重把组合预测还原到个股。
2. 记录这些组合先前的预测。对应的 5、20 或 60 步都已经过去后，从已观察的价格变化估算该笔预测的误差，剔除窗口内行情不完整的股票。每 48 步更新一次，各交易时段重新开始。第一个训练期滚动窗口选中用组合自身误差进行修正，在第二个窗口也获得正增益。该层推理只使用当前及此前行情和预测。

后四天的 Y1、Y2 沿用相同预测。下表中的变化均以上一列模型为基准。

| 交易日 | 市场共同修正 | 加入五个组合 | 再加入已兑现误差 | 组合层增量 | 误差层增量 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 17 | 0.408737 | 0.408867 | 0.408866 | +0.000130 | −0.000001 |
| 18 | 0.325073 | 0.324968 | 0.324965 | −0.000105 | −0.000003 |
| 19 | 0.359566 | 0.359819 | 0.359943 | +0.000253 | +0.000124 |
| 20 | 0.369997 | 0.370070 | 0.370053 | +0.000073 | −0.000017 |

按整日以及日内 120 步区块做 2,500 次配对重采样，相对市场共同修正版，两层合计增量的 95% 区间为 −0.000122 至 +0.000325。分别剔除一个验证日后，剩余三天的合并增量介于 +0.000027 和 +0.000174。四个验证日已多次用于模型比较。

完成上一节后，在同一 `src` 目录执行以下命令。两次训练分别保存本机参数文件，`verify` 完整回放 4 天，检查截断前缀，并重新加载参数复算。推理入口为 `released_model.SessionRefinement.predict(raw, base, opening)`。组合层与已兑现误差层使用同一份上游预测作为输入。

```bash
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -u experiment_y3_session_characteristic_portfolios_v1.py train
python -u experiment_y3_session_characteristic_portfolios_v1.py screen
python -u experiment_y3_session_characteristic_portfolios_v1.py verify
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 python -u experiment_y3_realized_sort_rotation_v1.py train
python -u experiment_y3_realized_sort_rotation_v1.py screen
python -u experiment_y3_realized_sort_rotation_v1.py verify
python -u audit_y3_session_market_uncertainty_v1.py --rotation --from-common
python -u audit_y3_characteristic_transfer_v1.py
python -u replay_session_market_release_v1.py --stage rotation --base-src /path/to/causal_16d/src --data project/data/public.parquet
```

[五个组合的训练与预测](src/experiment_y3_session_characteristic_portfolios_v1.py) · [已兑现误差的训练与预测](src/experiment_y3_realized_sort_rotation_v1.py) · [按股票和预测区间分解增益](src/audit_y3_characteristic_transfer_v1.py)

四天合并后，五个组合的综合分为 **0.3685044604285119**，加上盘中已兑现误差后为 **0.36852412142611474**，Y3 相关系数为 **0.3030828943303158**。两步相对上一节提高 **0.000095701978**，相对 GitHub 上一正式提交 `2e58a9d` 提高 **0.006117489873**。两份独立源码副本重训后，保留的两层参数数组分别逐项相同。

以市场共同修正版为基准，对 50 股逐时刻减去截面均值后，Y3 相关系数从 0.271952 升至 0.272227。第 21–60 步对应的远段截面相关系数从 0.119466 升至 0.119698。50 只股票中有 31 只的单股 Y3 相关系数提高，剔除任一只股票后，剩余股票合并的综合分增量介于 +0.000079 和 +0.000120。

## Y2 专用组合与最终推理入口

以 Y2 作为主目标，按收益的尺度换算 Y1，在训练中额外考察第 6–20 步的剩余收益。五个股票组合相对当下 Y2 预测截面重新构造，系数独立拟合。第 9–12 天和第 13–16 天，严格按日期截断的滚动对照中，Y2 相关系数分别提高 0.000144、0.000142。最终推理先根据同一组上游预测计算 Y2、Y3 的修正，再写入两列。

第 17–20 天的全部 936,250 行合并计算，三个目标相关系数依次为 **0.5010972595601481、0.3894525080116368、0.3030828943303158**，比赛综合分为 **0.36859665148067855**。Y2 专用头额外提高综合分 **0.000072530055**。本节与上一节的三个新增部分合计，相对市场共同修正版提高 **0.000168232032**，相对远端正式提交 `2e58a9d` 提高 **0.006190019928**。

| 交易日 | 最终综合分 | 相对仅调整 Y3 的增量 |
| --- | ---: | ---: |
| 17 | 0.408906 | +0.000040 |
| 18 | 0.325102 | +0.000137 |
| 19 | 0.360024 | +0.000080 |
| 20 | 0.370087 | +0.000034 |

与市场共同修正版相比，三个新增部分在四天的单日综合分都提高。配对重采样得到综合增量的 95% 区间约为 −0.000015 至 +0.000394，其中 Y2 单独贡献的加权增量区间约为 +0.000022 至 +0.000157。

完成前面各阶段后，在同一个 `src` 目录执行。固定 BLAS 线程数可复现本次重训的系数数组。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u experiment_joint_sort_horizon_transfer_v1.py train
python -u experiment_joint_sort_horizon_transfer_v1.py screen
python -u experiment_joint_sort_horizon_transfer_v1.py verify
python -u audit_joint_sort_transfer_v1.py
python -u replay_session_market_release_v1.py --stage joint --base-src /path/to/causal_16d/src --data project/data/public.parquet
```

完整模型使用 `released_model.SessionRefinement().predict(raw, base, opening)`。训练参数与回放报告位于本机 `artifacts/project_restart_generalization/joint_sort_horizon_transfer_v1/`。[Y2 的训练与验证](src/experiment_joint_sort_horizon_transfer_v1.py) · [完整推理与重算](src/replay_session_market_release_v1.py) · [Y2、Y3 综合增量的配对复核](src/audit_joint_sort_transfer_v1.py)

三层各自重新加载参数后，四天合并和逐日的全部分数均与各自的完整分阶段回放记录逐项相同。固定上述线程数从独立源码副本重训，三层的参数数组分别逐项相同。


## 用过去交易日的盘口特征估计个股反应

每只股票从历史交易日获得 15 个描述值，分别反映价格变动幅度、变动频率、价格自相关，以及四对盘口字段的差值大小、持续性和与随后一步价格变化的相关性。每个描述值都先在当日 50 股内标准化。按历史平均描述值找出 10 只接近的其他股票，利用它们在以往交易日对市场预测和盘口变化的反应，估计当前股票的 Y3 修正。将历史交易日交替分成两组，分别选邻居并估计系数，仅采用两组符号一致的部分，幅度取两者绝对值中较小者。

用第 9–12 天选择修正方式，在第 13–16 天确认。两组日期的严格按日期截断对照中，Y3 相关系数分别提高约 0.000067 和 0.000168。最终参数在第 17 天前冻结。正式比较使用前 16 天训练、后 4 天全部 936,250 行合并计算。综合分由上一节的 **0.36859665148067855** 升到 **0.36859841347694133**，Y3 相关系数为 **0.30308641832241145**。相对远端正式提交 `2e58a9d`，累计提高约 **0.00619178**。

本层的综合增量约为 +0.00000176。第 17、18 天上升，第 19、20 天下降。按交易日和 120 步区块配对重采样 2,500 次，增量的 95% 区间约为 −0.00008280 到 +0.00008338。使用独立重训的参数和上版由原始行情重建的上游预测，还原最终预测得到 **0.36859841346269934**。

在 `src` 目录完成上一节训练后运行以下命令。第一条重建训练期滚动预测并训练，第二条完整回放及重载参数复算，第三条从原始行情调用整套发布模型。单独复现这一节时，使用 `released_model.SessionRefinement(include_settled=False).predict(raw, base, opening)`。本层使用已经完成三个目标修正的预测作为输入。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u experiment_y3_peer_response_fingerprints_v1.py train
python -u experiment_y3_peer_response_fingerprints_v1.py verify
python -u replay_session_market_release_v1.py --stage peers --base-src /path/to/causal_16d/src --data project/data/public.parquet
```

[训练与预测](src/experiment_y3_peer_response_fingerprints_v1.py) · [训练期滚动与四天验证](src/y3_portfolio_walkforward.py) · [配对区块分析](src/audit_peer_response_upstream_v1.py)

## 用已经实现的 20 步预测误差修正 Y3

每个交易时段重新开始计时。每 96 步更新一次个股系数。回看最近 960 步，每隔 20 步取一个起点，要求完整的 20 步价格变化在本次更新前已经观测到。训练期拟合的换算关系将这些价格变化转换到 Y2 的尺度。对含缺失行情的个股跳过该段计算。推理只读取行情字段和上游预测。

对每只股票，计算过去 Y2 预测误差与当时六个输入的关系。六个输入是三个预测、两对盘口字段的差值和当期价格变化，均相对于当时 50 股做标准化。交替使用两组互不重叠的 96 步区块估计关系并取均值，再减去所有股票的均值。最后用历史日期训练的 6 × 6 权重矩阵将这些系数与当前六个输入组合，只修正每只股票相对于截面均值的 Y3。

在第 9–12 天选择了 20 步误差，在第 13–16 天复核。严格日期截断对照中的 Y3 相关系数增量依次为 **+0.000259** 和 **+0.000096**，共 7 天为正。一套较旧的参考模型在第二个窗口下降 0.000015。采用探索性筛选时以正式指标和严格截断的参考模型为主，随后冻结参数并在后四天评分。

前 16 天训练、后 4 天共 936,250 行全量回放的加权分为 **0.36864250256394965**，上一版本地模型为 **0.36859841347694133**。本层将 Y3 相关系数从 **0.30308641832241145** 提高到 **0.3031745964964281**。相对远端正式提交 `2e58a9d` 的 0.36240663155303376，累计增量为 **0.006235871010915883**。

独立重训得到相同参数。发布源码用已验证的上游行情预测重建最后几层，得到 **0.3686425025477474**。第 17–20 天的单日加权分变化依次为 **−0.00004476、+0.00013360、+0.00005562、+0.00003183**。以日期和 120 步区块配对重采样 2,500 次，增量的 95% 区间约为 **−0.00003212 到 +0.00014623**。26 只股票的单股 Y3 相关系数提高。逐一去掉任意一只股票或一个交易日，剩余样本的合并分增量都为正。

在上一节的相似股票模型训练完成后，从本版本的 `src` 目录运行下列命令。单独复现本节时，推理入口为 `released_model.SessionRefinement(include_attention=False).predict(raw, base, opening)`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u experiment_y3_settled_stock_gradient_v1.py train --primary
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u experiment_y3_settled_stock_gradient_v1.py verify --primary
python -u replay_session_market_release_v1.py --stage settled --base-src /path/to/causal_16d/src --data project/data/public.parquet
```

[训练与滚动推理](src/experiment_y3_settled_stock_gradient_v1.py) · [完整构建回放](src/replay_session_market_release_v1.py)

## 用盘口相似程度为历史误差加权

在上一节的 20 步误差修正基础上，再比较每只股票当前状态与其最近 960 步内已结算的历史状态。用训练好的 6 × 6 矩阵计算每个历史时点与当前时点的相似度。对相似度减去中位数，用中位绝对值加 0.25 缩放，再用双曲正切限制极端值。每条历史误差的权重取决于相似度及经过时间。数据不足四个历史区间时，该层的增量为零。截面中心化后将增量加到 Y3。

在第 9–12 天和第 13–16 天，相对上一节模型，严格日期截断对照的 Y3 相关系数分别增加 **0.000089** 和 **0.000037**。用于计算误差的行情窗口在当前 96 步区块开始前已经结束，在新的交易时段重新建立历史。

使用第 1–16 天训练并冻结参数，在第 17–20 天全部 936,250 行回放，三项目标相关系数为 **0.50109726、0.38945251、0.30318536**，加权综合分为 **0.36864788577584606**。相对仅用 20 步误差修正的模型再提高 **0.00000538**。相对相似股票版本的 0.36859841347694133，两项修正合计增加 **0.00004947**。相对 GitHub 正式版本 `2e58a9d` 的 0.36240663155303376，累计增加 **0.006241254223**。

重载参数后，四天全量回放的逐日分数和合并分数均与首次运行一致。独立重训复现了两层的参数。用上个版本已从行情重建的上游输出复核最终几层，得到 **0.36864788575939433**。相对相似股票版本，第 17–20 天的加权分变化依次为 **−0.00007574、+0.00015235、+0.00009168、+0.00003738**。29 只股票自身的 Y3 相关系数提高。按日期及 120 步区块配对重采样 2,500 次，两项修正合计增益的 95% 区间约为 **−0.00005321 到 +0.00017076**。

完成上一节训练后，从本版本的 `src` 目录执行下列命令。默认推理入口为 `released_model.SessionRefinement().predict(raw, base, opening)`。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u experiment_y3_settled_error_attention_v1.py train
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u experiment_y3_settled_error_attention_v1.py verify
python -u replay_session_market_release_v1.py --stage attention --base-src /path/to/causal_16d/src --data project/data/public.parquet
```

[训练与相似度计算](src/experiment_y3_settled_error_attention_v1.py) · [按日期和股票配对复核](src/audit_peer_response_upstream_v1.py)


## 盘口变化顺序与 Y3 · 0.36876656

这一层描述每只股票的压力和活跃程度变化的先后顺序。以 12、48、144 步为窗口，把两个字段的历史观测连成折线并闭合，计算带符号的面积。同样的起点和终点，若压力上升和活跃度上升的先后顺序相反，可以得到不同的面积。特征在午间重新起算。训练时控制窗口两端的变化及乘积，分别为个股相对其他股票的变化和市场共同变化拟合系数。选中的版本保留在训练日期的奇偶两个子集中方向相同的系数。

前 16 天用于训练。内部按日期前推，第 9–12 天选择候选，第 13–16 天复核。冻结参数后，在第 17–20 天全部 936,250 行计算相关系数。Y1、Y2、Y3 分别为 **0.5010972595613465、0.38945250801155423、0.30342271643654967**，加权分为 **0.3687665625340104**。相对误差注意力版增加 **0.00011867675816434886**，相对远端提交 `2e58a9d` 累计增加 **0.006359930980976647**。

重新加载参数后，四天完整回放的合并分及逐日分数与首次运行完全一致。独立目录重训得到一致的参数和内层筛选记录。按前一版冻结的上游预测复核，两版按天的加权分差依次为 **−0.00034901、+0.00069673、+0.00000206、−0.00005050**。50 只股票中 33 只的 Y3 相关性提高。按天及 120 步区块进行 2,500 次配对重采样，增益的 95% 区间为 **−0.00038195 到 +0.00064047**。去掉第 18 天，剩余三天的合并增益为 **−0.00014309**。

按前文完成历史各阶段训练后，从本版 `src` 目录执行以下命令。训练脚本会按日期重建并在本地缓存前一版的训练期预测。运行时，`released_model.SessionRefinement().predict(raw, base, opening)` 默认启用本层。

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u experiment_y3_ordered_book_area_v1.py train
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -u experiment_y3_ordered_book_area_v1.py verify
python -u replay_session_market_release_v1.py --stage area --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[训练与推理代码](src/experiment_y3_ordered_book_area_v1.py) · [训练期回放](src/y3_settled_walkforward_panels.py) · [按日期和股票复核](src/audit_peer_response_upstream_v1.py)

## 关联股票的压力与参与程度

前 16 个交易日训练，第 17–20 天共 936,250 行全量验证。新版本的 Y1、Y2、Y3 相关系数为 **0.5010972595613465、0.38945250801155423、0.3034349606779849**，加权综合分为 **0.36877268465472807**。相对本地盘口顺序模型增加 **0.000006122121**，相对上次 GitHub 正式提交 `2e58a9d` 的 0.36240663155303376 累计增加 **0.006366053102**。

第一层按交易时段计算 5 步和 20 步收益，减去同期 50 只股票的中位数。在估计的相关矩阵上加正则，再由逆矩阵计算偏相关。将过去交易日交替分为两组，保留两组中偏相关同号的股票联系，再按接收方筛去弱联系。推理时，以其他股票三个盘口压力字段的 6 步与 48 步均值之差构造信号。接收方近期价格已沿关联压力方向变动时下调信号，之后在 50 只股票间去均值并修正 Y3。这一层的综合分为 **0.36877207929118044**。

第二层用各关联股票贡献的绝对值和平方和，估计本次压力主要来自多少只股票。训练时先控制第一层已有的压力强度，再估计压力来源集中时应如何调整预测。午间重置历史窗口。两层均用第 9–12 天选择候选，在第 13–16 天复核，最后冻结参数。每个内层预测时点只利用此前训练日期拟合的股票关系和系数。

两个阶段各自从原始行情全量回放两次，每次重新加载参数。合并及逐日分数复现一致。独立目录用发布源码重训，两层的参数数组和训练期筛选记录均与实验目录一致。用独立重建的上游输出回放两层发布代码，综合分为 **0.3687726846399381**，截断行情的前缀预测最大误差为 8.34e-07。相对盘口顺序模型，四个交易日的综合分增量依次为 **-0.00000682、+0.00001520、-0.00000139、+0.00002136**。按交易日和 120 步时间块配对重采样 2,500 次，综合增量的 95% 区间为 **-0.00001294 至 +0.00002740**。在这四天，29 只股票自身的 Y3 相关性提高。

完成前文的盘口顺序模型训练后，在本版 `src` 目录依次执行下列命令。本地保留权重和预测缓存。完整模型入口为 `released_model.SessionRefinement().predict(raw, base, opening)`。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_unpriced_peer_pressure_v1.py train
python -u experiment_y3_peer_shock_participation_v1.py train
python -u experiment_y3_unpriced_peer_pressure_v1.py verify
python -u experiment_y3_peer_shock_participation_v1.py verify
python -u replay_session_market_release_v1.py --stage participation --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[股票联系与压力修正](src/experiment_y3_unpriced_peer_pressure_v1.py) · [压力来源集中度](src/experiment_y3_peer_shock_participation_v1.py) · [发布推理入口](src/released_model.py) · [逐日和逐股票配对复核](src/audit_peer_response_upstream_v1.py)

## 跨股票递归状态的分期集成

这一层只校正 Y3。用历史数据估计 50 只股票之间的联系，将每只股票当前的 100 个字段编码为递归状态，同时接收关联股票上一步的状态。真实联系与三组重新分配股票身份的联系共用输入和递归变换。两者的差值用于估计股票之间传递的信息。每个交易时段开头重置状态，所有输入截至当前时点。

在第 8、10、12、14、16 天结束时分别保留一套股票关系和读出参数。读出方向取两组交替交易日都支持的部分。预测时将五套参数各自的 Y3 修正取平均。前 16 天内部按日期向前检验时，第 9–12 天和第 13–16 天的 Y3 相关系数分别提高 0.000768 和 0.000429。

第 17–20 天共 936,250 行的合并分数为 **0.36878695676703543**，Y1、Y2、Y3 相关系数分别为 **0.5010972595613465、0.38945250801155423、0.3034635049025997**。相对上一本地版本提高 **0.00001427211230736**，相对 GitHub 已发布提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的 0.36240663155303376 累计提高 **0.00638032521400167**。第 17–20 天逐日综合分的变化依次为 +0.00058121、−0.00067606、+0.00008461、+0.00005747。按交易日和 120 步时间块配对重采样，综合分增量的 95% 区间约为 −0.000660 至 +0.000663。

先完成上节的股票压力和集中度阶段及其历史滚动预测，再在本版 `src` 目录执行以下命令。最后一行使用训练好的原版 Predictor，由验证集的行情列重建完整预测，并检查截短行情的前缀。单日预测形状为 `(时间步, 50, 3)`。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_echo_vintage_ensemble_v1.py train
python -u experiment_y3_echo_vintage_ensemble_v1.py verify
python -u replay_session_market_release_v1.py --stage echo-vintages --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[递归模型和训练入口](src/experiment_y3_echo_vintage_ensemble_v1.py) · [完整模型入口](src/released_model.py) · [验证入口](src/replay_session_market_release_v1.py)

## 用同类股票的盘口时滞修正 Y3

在每个时点按三个预测分别做截面尺度标准化，为每只股票找预测最接近的七只股票，再将配对关系对称化。网络取最近 64 步和 256 步两种窗口，各切成八段。它比较股票 A 的后段盘口与股票 B 的前段盘口，再交换 A 和 B，取两种匹配程度的差。训练标签是两只股票之间的收益残差之差。训练同时使用 Y3 和扣除前 20 步后的远段收益。每个时点聚合邻居的估计，扣除单股历史分量，得到对 Y3 的修正。模型在推理时只读取截至该时点的行情和上游预测。

训练使用前 16 天。第 9–12 天、第 13–16 天的滚动检验中，这一修正分别提高 Y3 相关系数约 0.0000060 和 0.0000102。最终模型在第 17–20 天合计 936,250 行上的综合分为 **0.36879276314072756**，Y1、Y2、Y3 相关系数分别为 **0.5010972595613465、0.38945250801155423、0.3034751176499839**。比上一节的模型高 0.00000580637，比 GitHub 提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 高 0.00638613159。四个单日相对上一节均有提升。按日期及 120 步时间块配对重采样，综合分增益的 95% 区间为 −0.00000006 至 +0.00001176。

先完成上一节的递归模型训练并准备其历史滚动预测。下面的命令在本版 `src` 目录运行。训练脚本在每次扩展训练集时重新拟合配对网络，验证脚本会重新加载模型并对比整日输入和截短输入的共同前缀。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_temporal_pair_alignment_v1.py train
python -u experiment_y3_temporal_pair_alignment_v1.py verify
python -u replay_session_market_release_v1.py --stage pair-alignment --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[配对模型训练](src/experiment_y3_temporal_pair_alignment_v1.py) · [组合预测入口](src/released_model.py) · [完整回放](src/replay_session_market_release_v1.py)

## 用已兑现的短期收益修正 Y3

模型把未来 60 步划分为前 5 步、第 6–20 步和第 21–60 步。一次预测发出后，每段分别在 5、20、60 步后到期。模型从已经观察到的行情重建各段收益，将每只股票的收益与同期股票平均收益比较。遇到缺失盘口模板时跳过受影响的收益区间。午间边界两侧分别计算。训练阶段沿用前文由早期交易日估计的行情到收益的换算参数。

每隔 240 步，模型用前 16 天积累的统计量和当天已经完整到期的区间，估计每只股票的预测需要怎样校准。它分别拟合三个期限，短期限只读取该期限及更短期限的预测，个股系数向 50 只股票共享的系数收缩。最后将个股特有的前 5 步、第 6–20 步两项调整相加，用于修正 Y3。当天收益的重建只需行情列。验证标签只在计算分数时使用。

前 16 天内的顺序检验选中了前 20 步的组合。第 9–12 天与第 13–16 天，Y3 相关系数分别提高约 0.000643 和 0.000260。第 17–20 天曾比较同时调整 Y2、Y3 的版本，Y2 相关系数降至 0.389077，因此最终沿用原 Y2 输出，只对 Y3 应用调整。这个输出选择也参考了后四天的验证结果。

第 17–20 天 936,250 行合并验证的综合分为 **0.3688925888703404**，Y1、Y2、Y3 相关系数分别为 **0.5010972595613465、0.38945250801155423、0.3036747691092096**。相对上一节配对模型，综合分提高 0.00009982573。四个单日综合分的变化依次为 +0.00016751、+0.00020024、−0.00009782、−0.00000185。对四天及日内 120 步时间块做配对重采样，综合分增益的 95% 区间约为 −0.000122 至 +0.000346。相对已发布提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的 0.36240663155303376，累计提高 0.00648595732。

先完成上述配对模型及训练期滚动预测，并准备行情到收益的换算参数。在本版 `src` 目录执行以下命令。`train` 训练本层并输出滚动选择结果，`verify` 读取原始行情做全量回放与参数重载验证。最后一条命令从发布版组合推理入口重新预测四天。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_early_settled_stock_slopes_v1.py train
python -u experiment_y3_early_settled_stock_slopes_v1.py verify
python -u replay_session_market_release_v1.py --stage early-stock-slopes --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[个股校准与训练入口](src/experiment_y3_early_settled_stock_slopes_v1.py) · [行情到收益的换算](src/experiment_compounded_return_audit_v1.py) · [组合预测入口](src/released_model.py)

## 按估计稳定性分配个股调整，午后迁移上午响应

训练集仍为前 16 天。这一层沿用上一节在前 20 步估计的个股修正，并使用两种信息决定修正的大小。

- 把历史日期分成两个交替子集，把当天已兑现收益的时间块也分成两个交替子集，交叉计算四组参数。比较它们对当前这一帧给出的调整。如果改变样本后某只股票的结果波动很大，就减小对这只股票的调整。在同一时点将保留的调整按截面均方根重新缩放，放大倍数上限为 2。
- 用历史交易日学习上午个股响应与下午响应之间的映射。每个训练日先用其他历史训练日计算参照值。推理时从行情中完整观测到上午后，才将该日的上午偏差映射到下午。随着下午已兑现的区间增多，上午信息的权重逐步降低。

第 9–12 天和第 13–16 天的滚动检验中，两项合用相对上一节分别提高 Y3 相关系数 0.00002758、0.00002789。第 17–20 天全部 936,250 行合并验证的综合分为 **0.36890193151512973**，Y1、Y2、Y3 相关系数为 **0.5010972595613465、0.38945250801155423、0.3036934543987883**。相对上一节的综合分增量为 0.00000934264，四天单日变化依次为 +0.00000211、+0.00004407、−0.00002636、+0.00001311。按日期和 120 步时间块做配对重采样，增量的 95% 区间约为 −0.00001883 至 +0.00004597。相对 GitHub 已发布提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的累计提升为 0.00649529996。

完成前文的短期收益校准阶段后，在本版 `src` 目录执行以下命令。`train` 重建历史滚动窗口与两组估计，`verify` 进行原始行情全量回放和参数重载对照。训练样本选择、模型参数和逐日验证指标均在本机生成。最后一行通过发布版预测入口重新计算后四天的结果。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_supported_morning_slopes_v1.py train
python -u experiment_y3_supported_morning_slopes_v1.py verify
python -u replay_session_market_release_v1.py --stage supported-morning --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[训练和推理修正](src/experiment_y3_supported_morning_slopes_v1.py) · [完整预测入口](src/released_model.py) · [逐日配对统计](src/audit_peer_response_upstream_v1.py)

## 用其他股票与近期价格变化解释 Y3 的跨股票修正

本层先估计每只股票当下的价格变化有多少可由过去和同帧其他股票解释，再用这部分历史变化修正 50 只股票的 Y3。三个步骤都在午间边界重新开始积累日内历史。

1. 用前一帧的 50 股价格变化及其 4 步和 20 步指数平均做岭回归，估计由此前行情解释的部分。用历史残差的正则化协方差估计同一帧其他股票提供的解释。估计的个股分量等于该股残差加上按精度矩阵加权的其他股票残差。从实际价格变化中扣除这个个股分量便得到本层采用的价格序列。
2. 对这条价格序列做 240 步指数衰减，并减去同帧 50 股均值。在训练期内，控制本股近 8 步、32 步的价格变化和三个已有预测后，估计各来源股票对各接收股票 Y3 误差的不同权重。修正以每个时点的股票截面均值为零。
3. 用较早日期的样本外预测估计修正幅度。第 1–4 天提供初始行情统计，第 5–16 天逐步加入训练，第 9–12 天选择候选，第 13–16 天检查重复性。最后用前 16 天可用记录重估参数，固定后预测第 17–20 天。Y1 和 Y2 沿用上一节的输出。

在前 16 天的两段滚动检验中，选中方案相对同一训练期参考预测的 Y3 相关系数分别提高 **0.00080069** 和 **0.00105566**。仅使用原始价格变化的对照分别提高 0.00054361 和 0.00056581。训练期参考预测来自当时可用的滚动预测面板。

在最后四天全部 936,250 行上，Y1、Y2、Y3 的相关系数分别为 **0.5010972595613465、0.38945250801155423、0.3037450848551476**，按 20%、30%、50% 加权后的综合分是 **0.3689277467433094**。相对前一节的本地模型提高 **0.00002581523**，相对 GitHub 已发布提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的累计提升是 **0.00652111519**。第 17–20 天单日综合分变化依次为 +0.00008862、−0.00027072、+0.00077429、−0.00030809。按日期与日内 120 步时间块进行 2,500 次配对重采样，综合分增量的 95% 区间为 −0.00042473 至 +0.00054082。发布源码独立重训得到相同的参数，用发布版组合预测模块重算四天得到 0.368927746724314。这四天已多次用于比较模型。

先按前文生成训练数据、滚动预测面板及上一节的参数，再从本版 `src` 目录执行。第一项命令重建本层使用的训练期预测面板，已有完整面板时可从第二项命令开始。第三项命令从原始行情做两次重载回放，最后一项命令通过组合预测入口复核。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u y3_settled_walkforward_panels.py
python -u experiment_y3_directed_unexpected_prices_v1.py train
python -u experiment_y3_directed_unexpected_prices_v1.py verify
python -u replay_session_market_release_v1.py --stage directed-prices --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[价格分解、时序训练与验证](src/experiment_y3_directed_unexpected_prices_v1.py) · [组合预测入口](src/released_model.py)

## 按价格反转调整历史记忆

这层只修正 Y3。先用前面已观测到的 50 股价格变化估计本股这一步的预期变动，再从实际变动中扣除预期，得到价格冲击。每只股票在 6、24、100 和 240 步四个尺度上各自积累冲击。当前冲击越大、方向与此前趋势越相反，模型就越快降低旧价格变化的权重。用每只股票对当帧截面统计量的影响表达其相对位置，再结合三个已有预测，拟合 Y3 的个股误差。所有时间窗口在午间重新开始。

前 16 天内，按日期滚动重训配对盘口网络、早段收益校准与价格响应层，得到本层实际使用的历史预测。初始比较中，附加方差归一化的候选在第 9–12 天增益最大，到第 13–16 天则弱于简单版本。两段训练回放均参与了最终选择。冻结后的版本采用反转加快衰减，Y3 在两段各提高 0.00167080 和 0.00056694。在独立工作目录从发布源码重建训练面板并重训，12 天历史预测和本层参数逐项相同。

前 16 天训练，后 4 天全量回放，936,250 行上综合分为 **0.3689484597859292**，Y1、Y2、Y3 相关系数为 **0.5010972595613465、0.38945250801155423、0.3037865109403872**。相对上一节本地版本的加权增量为 **0.00002071304**，相对 GitHub 已发布的 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 累计提高 **0.00654182823**。四天单日综合分变化依次为 −0.00062837、+0.00073858、+0.00013773、−0.00003909。按日期和 120 步时间块配对重采样的增量 95% 区间约为 −0.00055467 至 +0.00087616。四个日期已多次用于模型选择，这次小幅增益的跨日稳定性仍需新日期验证。

完成上一节的数据准备、训练和验证后，在本版的 `src` 目录执行以下命令。第一条 Python 命令生成训练期预测，第二条保存本层参数，第三条从原始行情进行全量回放及参数重载复验，最后一条通过发布版预测入口检查四天输出。数据、模型参数及逐日预测由本机生成。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u y3_directed_walkforward_reference.py
python -u experiment_y3_reversal_conditioned_memory_v1.py train
python -u experiment_y3_reversal_conditioned_memory_v1.py verify
python -u replay_session_market_release_v1.py --stage reversal-memory --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[逐日重建上游模型](src/y3_directed_walkforward_reference.py) · [价格记忆训练与推理](src/experiment_y3_reversal_conditioned_memory_v1.py) · [组合预测入口](src/released_model.py)

## 用连续两个五步窗口的波动修正 Y3

模型在每只股票上取相邻、互不重叠的两个五步价格变化，把两次变化的绝对值相乘，用 50 步和 250 步的指数窗口分别平滑。一次孤立的大幅变动对这个量的影响较小，而连续两段都有较大变动时它会上升。模型再将这个波动量与盘口方向、最近价格方向和已有预测相乘。拟合时扣除普通波动幅度已能解释的部分，用前 16 天的滚动样本估计剩余部分与 Y3 总收益及远段收益的一致关系。每只股票只用当时及以前的数据，午间重新初始化。本层只修正 Y3。

第 9–12 天和第 13–16 天，按训练实验每五步取一步的评估方法，Y3 的滚动相关系数分别提高 0.00026829 和 0.00006214；其中第 21–60 步的收益贡献分别为 +0.00009553 和 +0.00011607。用当时冻结的系数补算两段的所有时点，两段 Y3 增量分别为 +0.00013589 和 +0.00007900，其中远段收益贡献是 −0.00002855 和 +0.00014036。按前 16 天训练、后 4 天合并全部 936,250 行验证，三项目标相关系数依次是 **0.5010972595613465、0.38945250801155423、0.3038404253096599**。按 20%、30%、50% 加权后综合分为 **0.3689754169705655**。相对上一节本地版本提高 **0.00002695718**，相对已发布提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 累计提高 **0.00656878542**。

原始数据回放与参数重载复算逐项一致。在独立目录运行本版训练源码重新得到相同的滚动统计和参数。经发布版预测头从已验证的上游预测重新计算，四天综合分为 0.36897541696。四天中有三天的单日综合分提高。对日期与日内 120 步时间块做配对重采样，增量的 95% 区间约为 -0.00004011 至 +0.00009498。这些日期已反复用于模型比较。

完成上一节的训练后，在本版的 `src` 目录运行。第一项从训练日期重建历史预测，第二项拟合本层并保存本地权重，第三项从原始行情做两次回放。最后一项通过发布版预测接口复核结果。依赖及原始数据路径见文首。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u y3_reversal_training_days.py
python -u experiment_y3_local_jump_variation_v1.py train
python -u experiment_y3_local_jump_variation_v1.py verify
python -u replay_session_market_release_v1.py --stage bipower --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[本层训练与回放](src/experiment_y3_local_jump_variation_v1.py) · [完整组合预测入口](src/released_model.py)

## 汇总五种窗口起点的波动估计

每五步形成一个价格变化窗口，其起点有五种选择。上一层在时点 t 使用刚结束的那一种窗口估计。这个版本取 t 至 t−4 这五个时点分别算出的估计并等权平均，用于调节当下的盘口与预测信号。各股的信号、当前时点及三个现有预测仍逐帧计算。午间重新初始化。拟合沿用上一层的回归系数。前 16 天内逐帧比较等权、近期加权、中位数及错配股票等方案，选中等权平均。第 9–12 天和第 13–16 天的全时点滚动检验中，Y3 相对上一层分别提高 0.00002232 和 0.00003302。

用前 16 天训练，在后 4 天共 936,250 行上合并计算，Y1、Y2、Y3 的相关系数为 **0.5010972595613465、0.38945250801155423、0.3038567070043444**。加权综合分为 **0.3689835578179078**。较上一节提高 **0.00000814085**，较上一次远端正式版本 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 累计提高 **0.00657692626**。原始行情回放与参数重载复算逐项一致。从独立工作目录运行本版源码得到了相同的方案选择和训练指标。用发布版预测头从先前已验证的上游预测重新计算，四天总分为 0.36898355780。相邻两版分数差按日期和 120 步时间块重采样的 95% 区间为 -0.00000134 至 +0.00001764。这些验证日期已多次参与比较。

完成上一节后，在 `src` 目录执行。训练命令在本地写入方案选择记录，推理时读取该记录和上一层的参数。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_bipower_phase_pool_v1.py train
python -u experiment_y3_bipower_phase_pool_v1.py verify
python -u replay_session_market_release_v1.py --stage phase-pool --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[五种起点的训练与回放](src/experiment_y3_bipower_phase_pool_v1.py) · [发布版预测入口](src/released_model.py)

## 五步价格变化幅度与 Y3

这一层逐股计算过去五步价格变动的绝对值，按 50 步和 250 步两种尺度计算指数移动平均。在每个时点，用这两个数调节当前盘口、近期价格变化，以及原有三个期限预测对 Y3 的影响。午间重新开始累计。价格波动的标准化尺度由最早四天拟合，最终回归及校正强度用前 16 天内的历史预测拟合。

训练期的逐日比较中，第 9–16 天有 8 天的 Y3 相关系数提高。第 9–12 天和第 13–16 天分别提高 0.00040170 和 0.00043733。将波动信号换成另一只股票的波动，或延迟 100 步使用，两段训练期的增益均缩小。近段和远段的贡献随日期变化，本版按完整 Y3 选择。

前 16 天训练，在第 17–20 天共 936,250 行合并计算。Y1、Y2、Y3 相关系数分别为 **0.5010972595613465、0.38945250801155423、0.30392908305950633**，加权综合分 **0.36901974584548874**。较上一节提升 **0.00003618803**，相对远端正式版本 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 累计提升 **0.00661311429**。原始数据完整回放与重新加载训练参数后的第二次回放逐项一致。

独立目录重新训练本层得到相同的训练期指标。使用已验证的上游预测，经发布版预测入口计算得到 0.36901974583。较上一节，四天单日综合分变化依次为 -0.00001982, +0.00028462, -0.00000801, -0.00004963。根据日期和 120 步时间块重采样，增量的 95% 区间为 -0.00019865 至 +0.00029408。这四天已多次用于模型比较。

先完成上一节，进入 `src` 目录运行。前两条 Python 命令依次进行训练和双遍回放，最后一条检查完整发布版的预测入口。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_absolute_block_motion_v1.py train
python -u experiment_y3_absolute_block_motion_v1.py verify
python -u replay_session_market_release_v1.py --stage absolute-motion --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[五步价格变动的训练及回放代码](src/experiment_y3_absolute_block_motion_v1.py) · [发布版预测入口](src/released_model.py)

## 两段盘口压力与市场预测

逐股平滑盘口买卖压力，比较当前、8 步前与 40 步前的方向和强度。模型按前后两段压力的一致程度给股票连续赋权，计算不同股票群体的短期、中期和远期预测差。训练时扣除当前市场预测和最近一段压力已经能解释的部分，用较早交易日拟合剩余差异对 Y3 的修正幅度。上午和下午分别初始化，每次预测只读取截至当前时点的数据。

训练期第 9–12 天和第 13–16 天的 Y3 相关系数增量分别为 +0.00014853 和 +0.00016127，八个交易日中有 6 天提高。训练选出的修正主要作用于同时刻的市场共同预测。按头 20 步和后 40 步分解，后四天合并的 Y3 相关系数增量分别有 +0.00004693 和 +0.00003738 来自这两段。

使用前 16 天训练，第 17–20 天完整验证共 936,250 行。Y1、Y2、Y3 的相关系数分别为 **0.5010972595613465、0.38945250801155423、0.30401338645254977**，按 20%、30%、50% 加权得到 **0.36906189754201046**。相对上节提高 **0.00004215170**，相对上一次 GitHub 正式版本 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 累计提高 **0.00665526599**。原始数据回放和参数重载后的第二遍回放逐项一致。独立目录重新训练本层得到相同参数和滚动验证记录，通过发布入口复验的分数为 0.36906189754。

较上节，后四天的单日综合分变化依次为 +0.00017261, -0.00024206, +0.00009809, +0.00015540。按日期和 120 步时间块重采样，综合分增量的 95% 区间为 -0.00017863 至 +0.00023730。这四天已用于多轮模型比较。

完成上一节后，在 `src` 目录运行以下命令。第一行限制线性代数线程数，后面依次训练新增模型、重载验证、从原始数据检查发布预测入口。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_pressure_cohort_projection_v1.py train
python -u experiment_y3_pressure_cohort_projection_v1.py verify
python -u replay_session_market_release_v1.py --stage pressure-cohort --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[训练和回放代码](src/experiment_y3_pressure_cohort_projection_v1.py) · [发布版预测入口](src/released_model.py)

## 按较早轨迹选择参照股票

把 50 只股票固定分为两组，每组 25 只。对每只股票，用 12 步前和 48 步前的短期预测排名及当时已观测的波动，给另一组的股票分配参照权重。两段历史距离的权重分别为 0.6 和 0.4，再平衡每只股票被选作参照的总权重。对照当前盘口、平滑盘口和近期价格变化与这些参照股票之间的差异，在已有预测和当期匹配所能解释的基础上，估计对 Y3 的增量。午间重置历史，开盘初期使用本节首个可见时点补足缺少的滞后值。

逐步扩展训练窗口，第 9–12 天和第 13–16 天的 Y3 相关系数分别比上一节提高 0.00002938 和 0.00000910。正式切分使用前 16 天训练、后四天全量验证。合并 936,250 行后，Y1、Y2、Y3 的相关系数分别为 **0.5010972595613465、0.38945250801155423、0.30401571646010217**，加权综合分为 **0.3690630625457867**。相较上一节提高 0.00000116500，相较上一次 GitHub 正式版本 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 累计提高 0.00665643099。完整回放与参数重载回放结果逐项一致。独立目录重新训练所得参数一致，发布入口复验分数为 0.36906306253。

按未来时间段拆分，对 Y3 相关系数的贡献为前 20 步 -0.00000441、后 40 步 +0.00000674。第 17–20 天的综合分变化分别为 -0.00000068, +0.00000524, +0.00000739, -0.00000932。按日期和 120 步时间块重采样，增量的 95% 区间为 -0.00000914 至 +0.00000958。这四天已多次用于模型比较。

完成上一节后，在 `src` 目录依次运行训练、重载验证和发布模型的回放入口。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_forecast_trajectory_coupling_v1.py train
python -u experiment_y3_forecast_trajectory_coupling_v1.py verify
python -u replay_session_market_release_v1.py --stage peer-trajectory --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[训练源码与重载验证](src/experiment_y3_forecast_trajectory_coupling_v1.py) · [预测入口](src/released_model.py)

## 按实时波动校准远期预测幅度

用截至当前的价格变动估计每只股票的波动水平，指数移动平均跨度为 75 步，午间重新初始化。在股票和原预测强度相同的条件下，分别估计考虑波动状态和合并波动状态时的平均绝对收益。对两者之比取对数并截在 ±0.65，用训练期滚动预测学得的系数 0.16727 调整 Y3 后 40 步的预测幅度。调整量在每个时点扣除 50 只股票的平均值。预测时仅使用当前及以前的观测。

训练使用前 16 天，验证使用第 17–20 天全部 936,250 行。Y1、Y2、Y3 相关系数分别为 **0.5010972595613465、0.38945250801155423、0.30404774920939137**，加权综合分为 **0.3690790789204313**。两段训练期滚动验证的 Y3 增量分别为 +0.00002950 和 +0.00002847。后四天综合分比上节提高 0.00001601637，比上一次远端正式版本 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 累计提高 0.00667244737。原始数据回放和参数重载回放逐项一致，独立目录重新训练得到相同参数，发布预测入口复验综合分为 0.36907907890。

第 17–20 天的综合分增量依次为 +0.00008424, -0.00005974, -0.00004261, +0.00004941。对 Y3 相关系数的增量，未来前 20 步贡献 +0.00010910，后 40 步贡献 -0.00007707。对日期和 120 步时间块重采样 2,500 次，综合分增量的 95% 区间为 -0.00006581 至 +0.00009927。这四天已多次用于模型比较。

运行完上节的训练流程后，在 `src` 目录中执行下列命令。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_incremental_magnitude_ratio_v1.py train
python -u experiment_y3_incremental_magnitude_ratio_v1.py verify
python -u replay_session_market_release_v1.py --stage magnitude-ratio --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[训练源码与参数重载验证](src/experiment_y3_incremental_magnitude_ratio_v1.py) · [预测入口](src/released_model.py)

## 估计未来高波动时段的占比

在上节的幅度校准基础上，根据每只股票当前波动距离训练期中位数的程度，以及本次高波动或低波动状态已持续的步数，估计未来第 21–60 步处于高波动状态的比例。训练标签由前 16 天内已经结束的完整窗口计算。预测时只使用当前及更早的观测，并在午间和每日开盘重置计时。按估计占比在高、低波动幅度表之间插值，分别计算近段和远段相对沿用当前状态的预测差值，再对两段之和施加滚动校准系数 −0.10604。

前 16 天训练、第 17–20 天合并验证，共 936,250 行。Y1、Y2、Y3 的相关系数为 **0.5010972595613465、0.38945250801155423、0.30407356221312937**，加权综合分为 **0.36909198542230026**。相对上节提高 0.00001290650，两步幅度校准相对轨迹匹配版累计提高 0.00002892288。相对远端正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的累计增量为 0.00668535387。

训练期两段滚动验证的 Y3 增量分别为 +0.00001312 和 +0.00002131。后四天的新增 Y3 相关系数中，前 20 步贡献 -0.00000672，后 40 步贡献 +0.00003253。相对上节，第 17–20 天综合分分别变化 +0.00001807, +0.00004006, -0.00001904, -0.00000560。对日期和 120 步时间块重采样 2,500 次，两步合计综合分增量的 95% 区间为 -0.00006522 至 +0.00011618。这四天已多次用于版本比较。全量回放与重载结果逐项一致，独立目录重新训练所得参数一致，发布预测入口复验分数为 0.36909198540。

完成上节后，在 `src` 目录执行下列命令。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_expected_volatility_occupancy_v1.py train
python -u experiment_y3_expected_volatility_occupancy_v1.py verify
python -u replay_session_market_release_v1.py --stage expected-occupancy --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[训练源码与重载验证](src/experiment_y3_expected_volatility_occupancy_v1.py) · [预测入口](src/released_model.py)

## 根据每只股票已经兑现的短期预测误差修正 Y3

在每个交易时段内，分别回看 8、15、30 步。对每个回看长度，比较当前 Y1 预测与当时的 Y1、Y2 预测，得到六组变化。同时以 F0 重建截至当前的相对价格变化，将各股票的信号减去同期均值。五步实际价格变化兑现后，用过去的信号和对应的五步预测误差估计每只股票的反馈系数。估计采用 200 步指数平滑、方差正则化与有界系数，开盘及午后重新初始化。将价格和预测共同构造的六组反馈取均值，再减去仅由预测变化构造的反馈均值。训练期按历史日期校准后，以 0.36110561 倍将该差值加入 Y3。Y1 和 Y2 继续沿用上一版输出。

前 16 天训练，第 17–20 天合并验证，共 936,250 行。Y1、Y2、Y3 相关系数为 **0.5010972595613465、0.38945250801155423、0.305115813175679**，加权综合分为 **0.36961311090357507**。相对上一版本地验证分提高 0.00052112548，相对远端正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的累计增量为 0.00720647935。

第 9–12 天与第 13–16 天的滚动验证中，Y3 分别提高 0.00075306 和 0.00043413。第 17–20 天新增的 Y3 相关系数中，前 20 步贡献 +0.00074774，后 40 步贡献 +0.00029451。四天各自的综合分增量为 -0.00004352, +0.00226154, -0.00006178, -0.00035147。对日期和 120 步时间块进行 2,500 次配对重采样，综合分增量的 95% 区间为 -0.00047628 至 +0.00218238。这四个验证日已用于多轮版本比较。原始数据回放与重新加载参数的结果逐项相同，独立源码目录从训练到推理复核的综合分为 0.36961311088。

完成上一节的训练后，在 `src` 目录执行以下命令。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_pooled_revision_feedback_v1.py train
python -u experiment_y3_pooled_revision_feedback_v1.py verify
python -u replay_session_market_release_v1.py --stage revision-feedback --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[训练与验证源码](src/experiment_y3_pooled_revision_feedback_v1.py) · [预测入口](src/released_model.py)

## Y2 的独立反馈读出

固定上节的 Y3 模型。对 Y2 单独使用价格与预测共同变化所得的六组反馈均值，权重 0.69733226 由前 16 天的按日期滚动预测估计。Y3 使用其中超过单纯预测变化的部分。先计算两个预测所需的历史信号，再输出修正后的 Y2，避免让新 Y2 反向改变冻结的 Y3。Y1 沿用原模型。

前 16 天训练，第 17–20 天合并验证，共 936,250 行。Y1、Y2、Y3 的相关系数为 **0.5010972595613465、0.3911275521395039、0.305115813175679**，加权综合分为 **0.37011562414196**。本节相对上节提高 0.00050251324，相对正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的累计提升为 0.00770899259。第 9–12 天与第 13–16 天的滚动验证中，Y2 分别提高 0.00513347 和 0.00138748，八个验证日均为正增量。保持逐日系数不变，在未用于估计系数的其余 1,497,950 条同日期记录上重新检查，两段增量为 0.00527540、0.00152765。

第 17–20 天，本节新增的单日综合分依次为 +0.00001938, +0.00123366, -0.00005341, +0.00077675。日期和 120 步时间块配对重采样 2,500 次，综合分增量的 95% 区间为 -0.00010777 至 +0.00145821。两次反馈更新合计比本轮起点高 0.00102364，配对重采样区间为 -0.00039461 至 +0.00354161。这四天已用于多轮比较。原始数据全量回放与参数重载逐项一致，发布目录从头训练得到相同参数，独立预测入口综合分为 0.37011562412。

完成上节后，在 `src` 目录执行以下命令。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y2_matured_feedback_readout_v1.py train
python -u experiment_y2_matured_feedback_readout_v1.py verify
python -u replay_session_market_release_v1.py --stage medium-feedback --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[训练与验证源码](src/experiment_y2_matured_feedback_readout_v1.py) · [预测入口](src/released_model.py)

## Y1 的价格反馈

每只股票分别比较当前价格与 8、15、30 步之前的价格，并减去同一时点的市场平均变化。五步价格变化兑现后，根据该只股票已观察到的短期预测误差来校准三个回看长度的系数。取三路反馈的平均，以训练期滚动拟合的 0.39364971 倍加入 Y1。该读出在 Y2 和 Y3 更新之后执行，且所需预测输入只有原始 Y1。

前 16 天训练，第 17–20 天合并验证，共 936,250 行。Y1、Y2、Y3 的相关系数为 **0.5015221729322051、0.3911275521395039、0.305115813175679**，加权综合分为 **0.3702006068161317**。本节相对上节提高 0.00008498267，相对正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的累计提升为 0.00779397526。第 9–12 天与第 13–16 天的滚动验证中，Y1 分别提高 0.00093215 和 0.00053290。对照实验在相同回看长度下将系数移给另一只股票，或在各股票上使用固定系数，这两种设置在两个窗口的增益均低于个股系数。

相对上一版本，第 17–20 天的综合分依次变化 +0.00004920, +0.00015700, -0.00002409, +0.00014432。按交易日和 120 步时间块配对重采样 2,500 次，综合分增量的 95% 区间为 -0.00002813 至 +0.00021991。这四天已用于多轮比较。原始数据的全量回放和参数重载结果逐项一致，独立目录重新训练得到相同参数，发布预测入口复算综合分为 0.37020060679。

完成上节后，在 `src` 目录执行以下命令。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y1_price_feedback_v1.py train
python -u experiment_y1_price_feedback_v1.py verify
python -u replay_session_market_release_v1.py --stage short-price-feedback --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[训练与验证源码](src/experiment_y1_price_feedback_v1.py) · [预测入口](src/released_model.py)

## Y3 的多步波动比值

对于每只股票，用最近的价格计算相对 50 股平均走势的 5 步和 20 步价格变化。分别除以步数的平方根，再用 160 步的指数加权平均估计两种尺度的方差。只使用当时可见的价格，分别在上午和下午开盘重置历史。每只股票依照前 16 天估计的阈值将方差比划入低、中、高三个区间。

在相同股票、单步波动状态和预测强度下，估计方差比区间对应的未来收益绝对值。按这个估计相对基础表格的比值调整 Y3 的预测幅度，再减去同时刻全部股票调整量的均值。调幅使用训练期滚动估计的系数，应用在 Y1、Y2 的最后两项更新之前。

前 16 天训练，第 17–20 天全量验证，共 936,250 行。Y1、Y2、Y3 的相关系数为 **0.5015221729322051、0.3911275521395039、0.3051348669162197**，加权综合分为 **0.37021013368640204**。相对上节的综合分提高 0.00000952687。相对 GitHub 上一次正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的累计提升为 0.00780350213。第 9–12 天和第 13–16 天两个滚动窗口内，Y3 分别提升 0.00040057 和 0.00025996。

第 17–20 天的综合分增量依次为 +0.00013129, -0.00016246, -0.00000516, +0.00005617。按交易日和 120 步时间块配对重采样 2,500 次，综合分增量的 95% 区间为 -0.00028476 至 +0.00026679。这四天已用于多轮模型比较。原始数据回放与重载参数复算逐项相同，独立目录重新训练并经发布入口得到综合分 0.37021013360。

完成上节后，在 `src` 目录执行以下命令。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_variance_ratio_magnitude_v1.py train
python -u experiment_y3_variance_ratio_magnitude_v1.py verify
python -u replay_session_market_release_v1.py --stage variance-ratio-magnitude --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[训练与验证源码](src/experiment_y3_variance_ratio_magnitude_v1.py) · [预测入口](src/released_model.py)

## Y1 和 Y2 的波动比值

在上节的 Y3 更新后，分别为 Y1、Y2 估计当前收益预测对应的未来绝对收益。Y2 使用同只股票 20 步与 5 步的相对价格方差比；Y1 进一步减去同一时刻 50 只股票的平均对数方差比。两项估计使用各自最终短期预测、相应目标的预测强度分组，并通过较早日期拟合的系数调整预测幅度。两个目标的训练期筛选先于后四天评估，最终共同加入预测流程。

前 16 天训练，第 17–20 天全量验证，共 936,250 行。Y1、Y2、Y3 的相关系数为 **0.501618168935261、0.3912031966420404、0.3051348669162197**，加权综合分为 **0.3702520262377742**，相对上节再提高 0.00004189255。相对 GitHub 正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 的累计提升为 0.00784539468。第 9–12 天和第 13–16 天的训练期滚动验证中，Y1 分别提升 0.00015233、0.00008228，Y2 分别提升 0.00060416、0.00059068。

第 17–20 天的综合分增量依次为 +0.00006724, +0.00004174, -0.00004718, +0.00004657。按交易日和 120 步时间块配对重采样 2,500 次，综合分增量的 95% 区间为 -0.00008011 至 +0.00016886。这四天已用于多轮模型比较。原始数据回放与重载参数复算逐项相同，独立目录重新训练并经发布入口得到综合分 0.37025202617。

完成上节后，在 `src` 目录执行以下命令。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_short_medium_variance_ratio_v1.py train
python -u experiment_short_medium_variance_ratio_v1.py verify
python -u replay_session_market_release_v1.py --stage short-medium-ratio --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[训练与验证源码](src/experiment_short_medium_variance_ratio_v1.py) · [预测入口](src/released_model.py)

## 根据已兑现收益调整市场方向预测

从前 16 天各股票的中期预测中减去同期均值，用主成分表示 50 只股票之间的分歧。每个交易时段独立估计这些分歧与市场共同走势的关系。第一次更新使用已走完第 21–60 步的历史预测，第二次更新使用已走完 60 步的实际价格变化，估计增加这一信号对当前模型相关系数的作用。两步均只使用截至当前时刻的市场特征。预测入口在第 17–20 天不读取标签。校准系数使用前 16 天数据拟合。

前 16 天训练，第 17–20 天全量验证，共 936,250 行。Y1、Y2、Y3 的相关系数为 **0.501618168935261、0.3912031966420404、0.3052198915218845**，加权综合分为 **0.37029453854060657**。相对上一节的综合分增加 0.00004251230，相对上一次 GitHub 正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 累计增加 0.00788790699。在前 16 天内按时间滚动评估，第 9–12 天选中的市场调整使 Y3 相关系数增加 0.00009990，第 13–16 天同项增加 0.00051982。

第 17–20 天综合分逐日增量为 +0.00010660, -0.00001251, -0.00000870, +0.00005420。按交易日和 120 步时间块配对重采样 2,500 次，增量的 95% 区间为 -0.00002562 至 +0.00015271。这四天已用于多轮版本比较。全量回放与参数重载结果逐项一致，独立目录重训此校准模型后，发布预测入口得到综合分 0.37029453852。

完成上一节训练后，在 `src` 目录执行以下命令。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_prequential_disagreement_gate_v1.py train
python -u experiment_y3_prequential_disagreement_gate_v1.py verify
python -u replay_session_market_release_v1.py --stage market-disagreement-gate --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

[训练代码](src/experiment_y3_prequential_disagreement_gate_v1.py) · [预测入口](src/released_model.py)

## 根据盘中已兑现和部分兑现的收益校准 Y3

前 16 天用于训练，第 17–20 天全量验证，共 936,250 行。Y1、Y2、Y3 的相关系数分别为 0.501618168935261、0.3912031966420404、0.3056910985314719，加权综合分为 **0.3705301420454003**。相对上一节发布候选增加 0.00023560350。相对最近一次 GitHub 正式版本 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 累计增加 0.00812351049。

训练时，用此前日期拟合的参数逐步重放第 5–16 天的预测。每隔 5 步取一次样本拟合校准层，历史输入在每个时间步均更新。盘中用已走完 60 步、报价完整的样本估计历史预测修正与实际收益的关联。在两个互不重叠的时间块组中分别估计方向，方向一致时采用较小的系数。午休后重新累计。第二层等到历史预测已经走完 40 步，用已经观察到的这 40 步收益，加上当时可获得的未来 20 步预测来估计完整的 60 步结果，再更新预测修正系数。第 17–20 天的推理只读取截至当前时刻的特征。

第 9–12 天和第 13–16 天的滚动检验中，第一层使 Y3 相关系数分别提高 0.00034875 和 0.00012452，第二层进一步提高 0.00033274 和 0.00009499。第 17–20 天，单独使用第一层的综合分为 0.37047885691639165。第二层在此基础上增加 0.00005128513。第二层加权综合分逐日增量为 -0.00000742, +0.00024523, -0.00004008, -0.00001377。按日期及 120 步时间块配对重采样 2,500 次，综合分增量的 95% 区间为 -0.00008365 至 +0.00030585。两层合计在第 17–20 天的综合分增量为 -0.00000091, +0.00101427, -0.00013072, -0.00004616，对应重采样区间为 -0.00016308 至 +0.00114790。增益主要来自第 18 天，这四天已用于多轮版本比较。全量回放和参数重载结果一致，独立目录重新训练后通过发布入口得到综合分 0.37053014207。

完成上一节的上游训练后，在本版本的 `src` 目录执行以下命令。中间预测和训练参数写入本地 `artifacts` 目录。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u build_y3_dense_training_tape_v1.py --release-src . --output artifacts/project_restart_generalization/y3_dense_training_tape_v1
python -u experiment_y3_intraday_vintage_orientation_v1.py train
python -u experiment_y3_partial_maturity_orientation_v1.py train
python -u experiment_y3_partial_maturity_orientation_v1.py verify
python -u replay_session_market_release_v1.py --stage partial-maturity --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

## 从已兑现的组合误差学习跨股票预测修正

用前 16 天的历史数据确定 50 只股票中期预测的五个主成分。盘中把预测的变动投影到这些组合，再用已经走完 60 步的现金收益估计每个来源组合对各目标组合的预测误差有多少解释力。分别在交替的时间块上估计系数，方向相同时采用绝对值较小的估计。将组合修正还原为 50 只股票各自的 Y3 修正量，午休后单独累计。上游沿用第 1–8 天输入特征确定的缺失行模板，价格到目标的换算参数取自第 1–4 天。滚动比较使用第 9–16 天。后四天推理仅用当前及历史行情输入。

完成上一节训练后运行以下命令。第 1–16 天用于训练，第 17–20 天全量验证，共 936,250 行。三个相关系数为 0.501618168935261、0.3912031966420404、0.3057424625749854，加权综合分为 **0.37055582406715704**。相对上一节提高 0.00002568202，相对上次 GitHub 正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 累计提高 0.00814919251。

第 9–12、13–16 天的滚动检验中，Y3 相关系数分别提高 0.00205303、0.00008055。最后四天逐日综合分增量为 +0.00001639, +0.00008853, +0.00001040, +0.00003908。按交易日及 120 步时间块配对重采样 2,500 次，增量的 95% 区间为 -0.00009290 至 +0.00016731。相对本轮开始时的市场分歧模型，这三个新校准层合计提升 0.00026128558，其中第 18 天提升 0.00110279。合计增量的重采样区间为 -0.00022164 至 +0.00130025。这些验证日期已用于多轮版本比较。全量回放与参数重载结果一致，独立训练和发布预测入口复现综合分 0.37055582411。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_matured_revision_factor_transport_v1.py train
python -u experiment_y3_matured_revision_factor_transport_v1.py verify
python -u replay_session_market_release_v1.py --stage revision-transport --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

## 用两组股票的预测变化校准第三组

将 50 只股票按固定种子分为三组，轮流选择其中一组作为预测对象。另外两组分别提供 8 步内经价格调整的中期预测变化。训练期间对两个来源组的协方差做正则化典型相关分析，提取四组对应方向。盘中用已经走完 60 步的实际价格变化估计每个来源组与目标组个股预测误差的关系。用第一组估计的系数处理第二组的信号，再反过来计算一次，取两次修正的平均值。每个交易时段独立累计，预测只使用截至当前时点的输入。

完成上一节训练后运行以下命令。第 1–16 天用于训练，第 17–20 天全量验证，共 936,250 行。三个相关系数为 0.501618168935261、0.3912031966420404、0.3058941433933617，加权综合分为 **0.37063166447634516**。相对上一节提高 0.00007584041，相对上次 GitHub 正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 累计提高 0.00822503292。

第 9–12、13–16 天的滚动检验中，Y3 相关系数分别提高 0.00067268、0.00074314。第 9 至 16 天逐日均有提升。第 17–20 天逐日综合分增量为 -0.00004252, +0.00028029, +0.00015349, -0.00007449。按交易日及 120 步时间块配对重采样 2,500 次，增量的 95% 区间为 -0.00015847 至 +0.00035442。把四个方向分别加回上一节模型，并按所有加入顺序平均，得到它们对后四天 Y3 相关系数的贡献分别约为 −0.000119、+0.000140、+0.000108 和 +0.000024。第一方向与来源组平均变化的平方相关系数约为 0.96。此项分解在参数冻结后进行，模型训练保留了全部四个方向。这些验证日期已用于多轮比较。全量回放与参数重载结果一致，独立训练和发布预测入口复现综合分 0.37063166450。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u build_y3_revision_transport_training_tape_v1.py
python -u experiment_y3_independent_peer_sensors_v1.py train
python -u experiment_y3_independent_peer_sensors_v1.py verify
python -u replay_session_market_release_v1.py --stage peer-sensors --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

## 五步预测的跨股票修正

固定三个互不重叠的股票组，用两个来源组分别预测第三组各只股票的相对收益。每个来源组使用经已发生价格变化调整的两步 Y1 预测修正。将来源信号映射为四个方向后，用已经走完五步的历史收益估计来源方向与接收股票误差的关系。每个来源组使用自己估出的系数，再对两个来源组的预测取平均。通过交替时间块检查估计符号是否一致，午休后重新累计。

完成上一节训练后运行以下命令。第 1–16 天用于训练，第 17–20 天全量验证，共 936,250 行。三个相关系数为 0.5016697660302074、0.3912031966420404、0.3058941433933617，加权综合分为 **0.3706419838953344**。相对上一节提高 0.00001031942，相对上次 GitHub 正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 累计提高 0.00823535234。

第 9–12、13–16 天的滚动检验中，Y1 相关系数分别提高 0.00020232、0.00015309，八个日期有七天提高。第 17–20 天逐日综合分增量为 +0.00001202, +0.00004831, -0.00000737, -0.00001925。按交易日及 120 步时间块配对重采样 2,500 次，该版增量的 95% 区间为 -0.00001730 至 +0.00004161。这些验证日期已用于多轮比较。全量回放与参数重载结果一致，独立训练和发布预测入口复现综合分 0.37064198391。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u build_short_reference_for_peer_sensors_v1.py
python -u experiment_y1_source_specific_peer_response_v1.py train
python -u experiment_y1_source_specific_peer_response_v1.py verify
python -u replay_session_market_release_v1.py --stage fast-peer --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

## 已兑现的中期预测误差与 Y3

在每个时点读取 20 步前作出的 Y2 预测，将它换算为 Y3 的收益尺度，与这 20 步已经发生的价格变化比较。每个交易时段重新开始累计，跳过跨缺失行的窗口。对 50 只股票的误差作三种速度的指数平滑。用训练日的协方差为每只股票估计来自其余 49 只股票的权重。去除同组股票的均匀加权部分后，按组内相对变化修正各股 Y3。最后在截面上去均值并使用训练期估计的修正幅度。

完成上一节训练后运行以下命令。第 1–16 天用于训练，第 17–20 天全量验证，共 936,250 行。三个相关系数为 0.5016697660302074、0.3912031966420404、0.3059059998401959，加权综合分为 **0.3706479121187516**。相对上一节提高 0.00000592822，相对上次 GitHub 正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 累计提高 0.00824128057。

第 9–12、13–16 天的滚动检验中，Y3 相关系数分别提高 0.00116409、0.00086552，八个日期有六天提高。两个窗口都覆盖全部时点，在五种步长为 5 的抽样起点上均有提高。第 17–20 天逐日综合分增量为 -0.00013075, +0.00011871, +0.00006117, +0.00002271。按交易日及 120 步时间块配对重采样 2,500 次，该版增量的 95% 区间为 -0.00040186 至 +0.00044218。这些验证日期已用于多轮比较。全量回放与参数重载结果一致，独立训练和发布预测入口复现综合分 0.37064791209。

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_peer_settled_medium_errors_v1.py train
python -u experiment_y3_peer_settled_medium_errors_v1.py verify
python -u replay_session_market_release_v1.py --stage peer-settlement --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

## 根据中期预测差距校准 Y3

在前 16 天内按日期滚动拟合。对同一时刻的股票对，按 Y2 预测差距分为五档，用已经兑现的 20 步相对价格变化估计各档的方向判断能力和变动幅度权重。推理时，每隔 5 步读取一组满 20 步的历史 Y2 预测，按差距查表，累计每只股票相对于其他 49 只股票的预期判断能力。每隔 10 步更新一次系数，午间和隔日重置。最后用去除市场均值的系数调整该股票 Y3 中较远期的预测部分。该层在推理中只读取预测历史和当前预测。

完成上一节训练后，在同一源码目录执行以下命令：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y3_forecast_matchup_difficulty_v1.py train
python -u experiment_y3_forecast_matchup_difficulty_v1.py verify
python -u replay_session_market_release_v1.py --stage matchup --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

第 9–12 天、第 13–16 天滚动检验的 Y3 相关系数分别提高 0.00015844 和 0.00017073。前 16 天训练、后 4 天全量验证共 936,250 行，Y1、Y2、Y3 的相关系数为 0.5016697660302074、0.3912031966420404、0.3063371811988026，加权综合分为 **0.3708635027980549**。综合分较上一节提高 0.00021559068，相对 GitHub 正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 提高 0.00845687125。第 17–20 天的单日综合分增量依次为 +0.00008598, +0.00045253, +0.00014155, +0.00020725。

从原始数据回放和参数重新加载得到相同结果。独立源码目录重新训练该层并通过发布预测入口得到 0.37086350276。按日期及 120 步时间块配对重采样 2,500 次，估计本次综合分增量的 95% 区间为 +0.00001233 至 +0.00045546。后四天已经用于多轮模型比较。

## 根据股票对的预测差距校准 Y2

使用上一节的方向判断能力和变动幅度估计方式，使用两个错开 2 步的历史取样起点，平均两种日内估计。每个时点只读取当前及更早的 Y2 预测，对 50 只股票分别估计与其余股票的预期可区分程度。在所有原有输出生成后，将股票相对于同时刻市场平均水平的 Y2 预测乘以该股票的校准系数。Y3 延用上一节的校准。

完成上一节训练后，在同一源码目录执行以下命令：

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -u experiment_y2_whole_forecast_matchups_v1.py train
python -u experiment_y2_whole_forecast_matchups_v1.py verify
python -u replay_session_market_release_v1.py --stage medium-matchup --base-src /path/to/causal_16d/src --data /path/to/public.parquet
```

第 9–12 天、第 13–16 天滚动检验的 Y2 相关系数分别提高 0.00100137 和 0.00047154。前 16 天训练、后 4 天全量验证共 936,250 行，Y1、Y2、Y3 的相关系数为 0.5016697660302074、0.3915204395866485、0.3063371811988026，加权综合分为 **0.37095867568143737**。综合分较上一节提高 0.00009517288，相对 GitHub 正式提交 `2e58a9de73091f140504c137318a61a5ce4aa1b2` 提高 0.00855204413。第 17–20 天的单日综合分增量依次为 +0.00033496, +0.00007153, -0.00014767, +0.00002238。

从原始数据回放和参数重新加载得到相同结果。独立源码目录重新训练该层并通过发布预测入口得到 0.37095867563。按日期及 120 步时间块配对重采样 2,500 次，估计本次综合分增量的 95% 区间为 -0.00017792 至 +0.00044382。与加入 Y2、Y3 两层校准之前的 0.3706479121187516 相比，两步合计提高 0.00031076356。同法联合重采样的增量区间为 -0.00002376 至 +0.00074125。后四天已经用于多轮模型比较。
