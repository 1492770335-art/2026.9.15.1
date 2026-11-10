# Phase 0 赛题与数据审计报告（PH0_DATA_AUDIT.md）

- 生成时间：2026-11-07（Asia/Shanghai）
- 执行模型：deepseek-flash（DeepSeek Harness，本会话实际路由，见 Notion 指令"实际调用模型必须保存"）
- 状态：**G0 通过**（行级全量审计完成，无未解释异常）
- 证据文件：
  - 审计脚本：`project_restart/src/audit/audit_data.py`（SHA256 `f38d652a086244c70625a4f982bcb70cc02d527a22daaea7f538548099f33578`）
  - 结果 JSON：`project_restart/reports/ph0_audit_results.json`
  - 压缩包逐文件哈希：`project_restart/reports/archive_file_sha256.txt`（68 个文件）

---

## 1. 原始档案与完整性（CONFIRMED）

| 项 | 值 |
|---|---|
| 原始压缩包 | `C:\Users\asus\Desktop\比赛\project.tar.gz`，380,875,068 B，gzip（magic `1F 8B`） |
| 压缩包 SHA256 | `299bb831c7ff57f719062fc7e0f940369c85bd2ce8ab846bb9dd5cf44f972042` |
| 压缩包内 public.parquet SHA256 | `e03503ec4d519c043a5b95c4dd6684f25d2636dd00a1d015ba3128b1d7818556` |
| 已解包副本 | `project_restart/original/project/`（tar 只读解出，68 个成员，逐文件 SHA256 已存档） |
| 早前解包目录 | `project/project/`，其 parquet SHA256 与压缩包内成员**完全一致** |
| 同目录 `project.tar.exe` | Windows MZ 可执行文件，SHA256 `24642e7a…`，**不是赛题原始包**；本流程不执行该文件，仅记录存在 |

结论：`project.tar.gz` 即 Notion 记录的原始上传包；数据文件字节级一致。

## 2. 赛题规则核验（源码行号引用，CONFIRMED）

来源：`project/project/README.md`、`script/evaluate.py`、`script/base.py`。

- **数据结构**（README L7–L16）：50 股票、20 交易日、4,681,350 行；date int8、time int16（每日从 0 开始）、code int8（0–49 跨日不变）、f0–f82 float16 标准化连续特征、f83–f99 离散 0/1/2、y0–y2 float32 标准化标签。
- **真实列名**（实测）：`f000`–`f099`（README 写 f0–f99，按数值编号排序即可，见 README L21 与 evaluate.py L44 的 `int(name[1:])`）。
- **输入接口**（README L73–L78，base.py L14）：每 tick `float32 [50,100]`，行固定对应 code 0–49，列对应 f0–f99；离散列以 float32 传入，embedding 需自行转整型；返回 `[50,1]`。
- **reset 语义**（README L75，evaluate.py L49–L50）：每交易日开始前对每个模型调用一次 `reset()`；跨日不得保留未来状态。
- **评估流式过程**（evaluate.py L33–L61）：按 date 过滤 → `sort(['time','code'])` → 特征排序取 float32 → `reshape(-1,50,100)` → 逐 tick 逐模型 `predict(current.clone())` → 形状必须 `(50,1)` → 全部 finite 否则报错（L57–L59）。
- **评分口径**（evaluate.py L61–L67，README L120–L126）：跨全部评估日期合并（pooled）Pearson；`denominator==0` 时报 invalid predict；**先对 c0/c1/c2 各自 round(…,4)，再 `round(0.2*c0+0.3*c1+0.5*c2, 4)`**（Python round 为银行家舍入）。内部比较保留未舍入值。
- **提交结构**（README L80–L89）：`commit/{base.py, network.py, conf/Model_{0,1,2}.json, model/Model_{0,1,2}.pth, description.md}`；配置的 `model` 必须是 network.py 中的类名；三目标可不同架构。
- **示例模型**：commit_v0 = SiLU MLP 100→128→128→1（network.py L9）；commit_v1 = 单层 GRU hidden=128、window=100，逐股票历史缓存 + 日初 `reset()`（network.py L23–L27）。两者仅为接口示例。
- **示例训练**（example/main.py L21、fit.py L10–L29）：直接读全量 parquet、`shuffle=False`、MSE、无日期切分 → 训练内分数不能当 OOS（与 Notion 记录一致）。

## 3. 全量行级数据审计（CONFIRMED，非 footer 推断）

polars 流式全扫描（4.68M 行 × 106 列，耗时 3.5 s），结果见 `ph0_audit_results.json`：

| 检查项 | 结果 | 判定 |
|---|---|---|
| 总行数 | 4,681,350（= README 申报） | ✅ |
| 唯一键 (date,time,code) | 4,681,350 = 总行数 | ✅ 无重复键 |
| 每 tick 行数/股票数 | 全部 93,627 个 tick 恰好 50 行、50 个不同 code | ✅ |
| 列 dtype | date Int8 / time Int16 / code Int8 / f000–f082 Float16 / f083–f099 Int8 / y0–y2 Float32 | ✅ 与 README 一致 |
| 全列 null | 0 | ✅ |
| 全浮点列 NaN / Inf | 0 / 0（逐值扫描） | ✅ |
| f083–f099 取值域 | 每列 {0,1,2}，n_unique=3 | ✅ |
| date / code 域 | 0–19 / 0–49 | ✅ |
| 原始文件日内 time 单调 | 20/20 日期 true | ✅ |
| 官方 sort(['time','code']) 后 code 序列 | 每 tick 严格 0..49 循环 | ✅ 与官方 [50,100] 输入构造一致 |

### 每日规模

| 项目 | 值 |
|---|---|
| 每日 tick 数 | 4681 tick ×13 天（date 0,2,3,4,8,9,10,12,13,14,16,17,18）；4682 tick ×7 天（date 1,5,6,7,11,15,19） |
| 合计 | 93,627 tick = 4,681,350 行 ✓ |
| 每日 time | 0..T-1 连续，无缺口 |

（每日 tick 数不相等是数据固有结构，非异常；训练按日切分不受影响。）

### 全局极值与分位数

| 列 | min | q01 | q50 | q99 | max | std |
|---|---|---|---|---|---|---|
| f000 | -51.50 | -3.326 | 0.00136 | 3.348 | 74.75 | 1.0 |
| f082 | -1.2197 | -1.2197 | 0.0191 | 2.2305 | 3.912 | 1.0 |
| y0 | -34.665 | -2.888 | 0.0039 | 3.183 | 88.297 | 1.0 |
| y1 | -24.203 | -2.646 | 0.0076 | 3.122 | 59.964 | 1.0 |
| y2 | -20.535 | -2.531 | -0.0083 | 3.087 | 50.474 | 1.0 |

- 与 Notion 首轮 footer 记录一致，本轮为**逐行验证**。
- 极值事件聚集：y0 全局最大 88.3、y1 59.96、y2 50.47 均出现在 **date 7**；y0 全局最小 -34.67、y1 -24.20、y2 -20.54 均出现在 **date 8**。**date 8 恰好位于 Fold A 验证集（8–11）内**，Fold A 的 Pearson 会被这些极端观测放大影响——折选型时必须看逐日/去极端敏感性，不能只看 pooled 值。
- 每日期标签均值/标准差详见 JSON（各日 std 在 0.66–1.29 之间波动，跨日非平稳是金融数据正常现象）。
- 处理原则：**不 winsorize、不改数据**；仅在模型层面做稳健性消融。

## 4. 环境确认（CONFIRMED）

| 项 | 值 |
|---|---|
| OS | Windows（PowerShell） |
| Python | 3.13.5（`C:\Users\asus\AppData\Local\Programs\Python\Python313\python.exe`） |
| 依赖 | numpy 2.5.4、polars 2.1.0、pyarrow 26.0.1、tqdm 4.70.1（pip 安装成功，网络可用）；torch 正在安装 CUDA cu126 轮子（后台任务） |
| GPU | NVIDIA RTX 3060 6GB（Laptop），driver 566.07，CUDA 12.7，空闲显存 ~6.0/6.1 GiB |
| 内存 | 15.4 GB |
| 磁盘 | C: 剩余 84.8 GB |

## 5. 待确认事项与风险（INFERRED / OPEN）

1. **标签预测窗口（horizon）未知**：官方材料未声明 y 对应的未来时点。按日切分无法形式化排除跨日标签交叠 → 冻结验证前必须做 gap=0/1 日隔离敏感性（Phase 1）。
2. **官方评分舍入语义**：evaluate.py 先舍入三个 corr 再算加权分；报告与复现均保留未舍入 raw 值，同时给出官方四位分数。
3. date 16–19 历史项目已多次暴露 → 仅作为冻结后附加检查，不得称严格盲测。
4. `project.tar.exe` 为无关 MZ 可执行文件，不执行、不纳入证据链。

## 6. 复现命令

```powershell
cd C:\Users\asus\Desktop\比赛
$env:PYTHONIOENCODING='utf-8'
python project_restart\src\audit\audit_data.py
# 输出: project_restart\reports\ph0_audit_results.json
tar -xzf project.tar.gz -C project_restart\original   # 只读副本
```

## 7. G0 结论

- 行数/唯一键/每 tick 50 股/NaN/Inf/类别域/排序/标签分布全部实测通过，无未解释异常。
- 评分与提交接口已按源码逐条核验。
- **Gate G0：PASS。** 下一步进入 Phase 1（评分器确定性单测 + 切分冻结 + 标签 horizon 敏感性），然后 Phase 2 基线（Ridge → 官方 MLP → 官方 GRU）。
