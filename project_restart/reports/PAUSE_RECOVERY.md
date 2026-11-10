# 阶段暂停与恢复手册（PAUSE_RECOVERY.md）

- 暂停时间：2026-11-07 ~20:20（Asia/Shanghai）
- 原因：用户安全暂停指令（停止新训练、保留全部产物）
- 状态：所有训练进程已终止；GPU 空闲；无文件删除；冻结基线完整。

## 一、进程状态（CONFIRMED）

- 全部后台任务已终止（job_kill 队列 pwsh-252ea3d7cdc044f7）；`Get-Process python` 为空；nvidia-smi 0% / 69MiB。
- 无孤儿进程、无正在写入的文件（torch.save 仅在目标训练完成时一次性写出，终止时不破坏已有文件）。

## 二、检查点完整性（CONFIRMED）

- **冻结基线 EXP-0005 strong-GRU（seed 42）**：
  - `experiments/EXP-0005/commit_grustrong_foldA/{base.py,network.py,conf/Model_0..2.json,model/Model_0..2.pth}`（3×356,373 B）
  - `experiments/EXP-0005/commit_grustrong_foldB/`（同上）
  - 6 个 pth 均已用 torch.load(weights_only=True) 验证可加载，keys = gru.*/output.*
  - 分数：FoldA 0.2228（c0 0.4518/c1 0.285/c2 0.0938）、FoldB 0.2531（c0 0.4473/c1 0.2847/c2 0.1564）
- 其余已完成实验产物均在 experiments/EXP-0001..0004（commit + results/integrity/decision）。
- EXP-0006 部分结果：`pause_snapshot.json`（MLP 3 seeds×2 folds 已得；GRU seed43 仅 y0 ckpt，y1/y2 与 seed44 需重跑）。

## 三、已完成 vs 待验证

已完成（CONFIRMED，均已过官方 Evaluator）：
1. Phase 0 审计（G0 PASS）、评分单测（G1 PASS）、切分冻结（TC-03）。
2. 基线：Ridge 0.1929/0.2051；官方 MLP 0.1913/0.2105；官方 GRU 0.2012/0.2176；strong-MLP 0.2163/0.2319；strong-GRU 0.2228/0.2531。
3. MLP seed 噪声 σ≈0.001（同折 >0.003 可判显著）。
4. GRU 训练/流式语义一致性（0.2012==0.2012）。
5. 关键发现：Δy_t 与当前特征的线性相关 ≈ −0.70~−0.72（y0/y1/y2 一致），水平值相关仅 0.07~0.17 → delta 监督 + 自回归重建是下一阶段最高价值方向。

待验证（PAUSED）：
- EXP-0006 GRU seeds 43/44（噪声下限的 GRU 半边）
- EXP-0007 GRU 窗口 K∈{16,64}
- EXP-0008 市场状态（strong-MLP 宿主）
- EXP-0009 类别 one-hot（strong-MLP 宿主）
- EXP-0010 y2 Pearson 混合损失（预注册；受 delta 发现影响需重估动机）
- EXP-0014 delta-head MLP（预注册 + 脚本就绪，最高优先级）

## 四、恢复执行步骤

```powershell
cd C:\Users\asus\Desktop\比赛
$env:PYTHONIOENCODING='utf-8'
# 1) EXP-0014 delta-head（最高优先级）
python project_restart\src\models\mlp_delta.py
# 2) EXP-0006 重跑（GRU seeds；MLP 部分冗余 8 分钟，可接受）
python project_restart\src\models\exp0006_seeds.py
# 3) 原队列剩余
python project_restart\src\models\gru_window.py
python project_restart\src\models\mlp_market.py
python project_restart\src\models\mlp_catenc.py
# 或使用队列脚本（BOM-UTF8）：
powershell -ExecutionPolicy Bypass -File project_restart\scripts\run_queue_0006_0009.ps1
powershell -ExecutionPolicy Bypass -File project_restart\scripts\run_queue2_0014.ps1
```

决策点（恢复后按证据推进）：
1. EXP-0014 若两折显著超过 0.2228/0.2531 → delta 方向晋级为主线，再做 GRU-delta 与混合损失。
2. 否则回到窗口/市场/类别单因素消融。

## 五、Notion 同步状态

- 新增「阶段暂停快照」记录：https://app.notion.com/p/4118093a7cba81d293d2e662b7d33047
- TC-06/07/09 已回置「待办」（暂停说明在阻断条件字段）。
- 总控页第 15 节为暂停快照摘要。
