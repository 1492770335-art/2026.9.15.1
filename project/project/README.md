# 图灵建模大赛

本比赛提供匿名化的高频订单簿特征，请基于给定数据集训练模型，预测三个不同噪声水平的标签。你可以探索RNN类、Transformer、GNN等各种时序, 截面神经网络模型, 并基于你的理解进行模型架构以及损失函数改进, 推荐尝试时序与截面联合建模、多任务学习、自监督学习等方向。

## 数据

可用数据位于 `data/public.parquet`，包含50支股票, 20个交易日, 共4,681,350行, 约500MB。

| 列 | 含义 | 类型 |
|---|---|---|
| date | 按时间顺序编号的交易日 | int8 |
| time | 每日从0开始的时间步 | int16 |
| code | 固定匿名标的编号0–49，跨日不变 | int8 |
| f0–f82 | 已标准化的连续特征 | float16 |
| f83–f99 | 取值为0、1、2的离散特征 | int8 |
| y0–y2 | 已标准化的预测目标 | float32 |

### 提示

1. 离散特征 `f83–f99` 的取值为 `0、1、2`，可作为 embedding 的类别 ID。
2. 文件中的特征列名为 `f0、f1、…、f99`，请按数值编号排列输入列。训练与评估的特征顺序必须一致，否则会导致预测错误，严重影响评分。
3. 可自行划分公开数据作为训练集和验证集。请注意金融数据具有极强的自相关性，随机划分样本容易导致严重过拟合，建议**按日期进行划分**。

## 项目结构

```text
project/
├── script/
│   ├── base.py
│   ├── network.py
│   └── evaluate.py
├── data/public.parquet
├── example/
│   ├── conf/
│   │   ├── MLP/Model_0.json、Model_1.json、Model_2.json
│   │   └── GRU/Model_0.json、Model_1.json、Model_2.json
│   ├── dataset.py
│   ├── fit.py
│   ├── main.py
│   ├── base.py
│   ├── network.py
│   └── __init__.py
├── commit_v0/                  # MLP提交样例
│   ├── base.py
│   ├── network.py
│   ├── conf/Model_0.json、Model_1.json、Model_2.json
│   └── model/
│       ├── Model_0.pth
│       ├── Model_1.pth
│       └── Model_2.pth
├── commit_v1/                  # GRU提交样例，目录结构与commit_v0一致
├── README.md
└── requirements.txt
```

## 运行示例

`example/` 提供完整训练示例，`script/evaluate.py` 提供评估工具；`commit_v0/`、`commit_v1/` 分别展示MLP、GRU的完整提交样例。

从 project 根目录安装依赖，再进入 example 运行。参考环境为 Python 3.11，依赖见 requirements.txt；使用GPU时请安装与你的CUDA环境匹配的PyTorch。

```bash
pip install -r requirements.txt
cd example
python main.py --config conf/MLP/Model_0.json
python main.py --config conf/GRU/Model_0.json
```

示例用于展示接口和大致的训练流程，并非统一的比赛基准, 具体训练方式你可以自行探索。你需要自行完成三个目标的训练、验证与模型选择。

## 模型接口

模型须继承 `BaseModel`，构造函数接收 `config`，并实现：

- `reset()`：跨日清空状态，无状态模型可使用空实现。
- `predict(x)`：用于评估中接受当前时间步的各股票特征float32张量 `[50,100]`，返回单个目标的预测张量 `[50,1]`。

输入行号固定对应code 0–49，列号对应f0–f99。后台将Parquet特征转为float32后传入模型。离散列在输入张量中仍以float32表示，如使用embedding请自行转化为整型处理。

## 提交

 请参照`commit_v0/` 和 `commit_v1/`的结构创建自己的 `commit/`，配置和权重分别放在 `conf/`、`model/` 子目录。

- `commit/base.py`：主办方提供的BaseModel标准接口。
- `commit/network.py`：模型架构具体实现。
- `commit/conf/Model_i.json`：文件编号与标签一致, 每份配置的 `model` 参数必须属于network.py中存在的类名，三个模型可以使用不同架构和参数。
- `commit/model/Model_i.pth`：对应三份配置的state_dict权重，文件编号与标签一致。

同时也请提交关于模型结构、训练方法、加载方式及额外依赖的简要说明, 汇总为一个`description.md`。


## 评分

### 提交自查

在 project 根目录运行，选择公开数据中的某些天，检查提交权重能否正确加载、流式预测及评分。

`--date` 支持单个或多个日期，不传时评估文件内全部日期。`--commit_dir` 指定提交目录。

例如评估公开数据最后五天：

```bash
python script/evaluate.py --data data/public.parquet --commit_dir commit_v0 --date 15 16 17 18 19
```

也可以直接调用：

```python
from script.evaluate import Evaluator

evaluator = Evaluator('data/public.parquet', 'commit_v0', 'cpu')
result = evaluator.evaluate(date=0)
result = evaluator.evaluate(date=[15, 16, 17, 18, 19])
```

公开数据分数仅用于自查，不作为正式排名, 正式后台评估会采用与 `evaluate.py` 完全相同的计算口径。

### 计分方式

主办方运行提交的模型权重，在完整隐藏评估集上分别计算三个目标的Pearson相关系数：

```text
Score = 0.2 * Corr_0 + 0.3 * Corr_1 + 0.5 * Corr_2
```

Corr 基于全部评估样本合并计算, Score 保留四位小数, 作为正式排名依据，得分越高，排名越高, 具体计算以主办方运行结果为准。若得分相同，则使用额外的OOS数据，进行比较, 确定最终排名。
