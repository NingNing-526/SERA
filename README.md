# SERA

Semantic Evidence Retrieval and Alignment：基于语义锚点检索与文本引导融合的多模态情感分析。

本项目将现有实验实现整理成统一入口，默认 **batch size 8、seed 12345、Top-K 8/8/16**。支持 CMU-MOSI、CMU-MOSEI 及五种消融配置。本仓库仅发布代码、配置和数据准备说明，不包含大模型生成的数据；所需文件见 `data/README.md`，历史参考校验信息见 `data/manifest.json`。不包含原始多模态特征、BERT/Gemma 权重、训练检查点和历史实验结果。

这是统一配置后的代码版本，不是对历史混合 batch、多种子筛选均值的重新命名。单种子运行结果不等于论文五种子均值；原始种子、历史设置及原工程未被修改。

## 1. 目录

```text
SERA/
  run.py                       # 检查、训练、测试统一入口
  configs/default.json         # 唯一默认配置
  src/model/sera.py            # SERA 主模型
  src/model/                   # 编码、Top-K、对齐、融合及回归
  src/engine/                  # 训练、验证和测试
  src/dataloader/              # 原始数据与离线语义缓存加载
  src/common/                  # 指标、随机种子、日志及检查点
  src/semantic/                # 原始离线生成提示词及实现
  src/scripts/                 # 离线描述生成和词元编码工具
  tests/smoke_test.py           # 合成数据端到端检查
  docs/IMPLEMENTATION.md        # 结构、数据格式和兼容说明
  requirements.txt
```

## 2. 环境

使用已安装项目依赖的 Conda 环境（以下环境名仅为示例）：

```bash
conda activate sera
cd SERA
python run.py --help
```

新建环境可使用 Python 3.10，安装与显卡驱动兼容的 PyTorch 2.1.2 后执行 `pip install -r requirements.txt`。下游训练读取离线语义特征，不需要安装或运行 Gemma；不要把下游的旧版 Transformers 当作大模型生成环境使用。

## 3. 数据准备

需要三个本地路径：

- BERT-base-uncased 目录，含配置、词表和模型权重。
- 对应数据集的 `unaligned_50.pkl`。
- 与原始样本 ID 和排列完全一致的 `gemma_semantic_tokens_v11_len256.pt`。

默认放到 `models/bert-base-uncased` 和 `data/mosi/`，也可以使用命令行绝对路径。数据和缓存必须来自可信来源：pickle/torch 文件加载可能执行代码。

下面以项目根目录为当前目录，使用本地相对路径（Bash 示例）；也可替换为自己的绝对路径。

```bash
export BERT=models/bert-base-uncased
export RAW=data/mosi/unaligned_50.pkl
export SEM=data/mosi/gemma_semantic_tokens_v11_len256.pt
```

PowerShell 可使用 `$env:BERT = "models/bert-base-uncased"` 等形式赋值，并在命令参数中使用 `$env:BERT`、`$env:RAW`、`$env:SEM`。

`run.py` 的资源相对路径以项目根目录为基准；离线脚本请从项目根目录执行，使用 `--help` 查看路径参数。本机配置可保存为 `configs/local.json` 并通过 `--config configs/local.json` 加载，该文件不会提交。

MOSEI 对应路径：

```bash
export RAW=data/mosei/unaligned_50.pkl
export SEM=data/mosei/gemma_semantic_tokens_v11_len256.pt
```

## 4. 检查和训练

先以 MOSI 为例，检查三个划分的数据 ID、张量形状和首批前向计算：

```bash
CUDA_VISIBLE_DEVICES=0 python run.py check --dataset mosi \
  --bert-path "$BERT" --raw-data "$RAW" --semantic-features "$SEM"
```

正式训练（只使用验证集选择模型，不自动测试）：

```bash
CUDA_VISIBLE_DEVICES=0 python run.py train --dataset mosi \
  --bert-path "$BERT" --raw-data "$RAW" --semantic-features "$SEM"
```

MOSEI 使用相应 RAW、SEM 路径，将 `--dataset mosi` 改为 `--dataset mosei` 即可。默认输出分别为 `runs/mosi_full_seed12345` 和 `runs/mosei_full_seed12345`。程序拒绝覆盖非空训练目录，重跑需指定新的 `--output runs/new_run`。本版没有断点续训入口。

训练设置：最大 200 轮，编码器学习率 1e-5，新模块学习率 1e-4，weight decay 1e-3，warm-up 5 轮，dropout 0.1；辅助权重从0.1线性衰减并截断至0.01；验证 Loss 早停 patience=25、最小轮数120、min_delta=0.0005。所有实际参数保存为运行目录中的 `config.json`。

## 5. 测试和输出

```bash
CUDA_VISIBLE_DEVICES=0 python run.py test --run-dir runs/mosi_full_seed12345
CUDA_VISIBLE_DEVICES=0 python run.py test --run-dir runs/mosei_full_seed12345
```

测试自动读取该目录的配置及 `checkpoints/best_Loss_model.ckpt`，不重新训练，不重新选择种子或模型。目录迁移后可用 `--bert-path`、`--raw-data`、`--semantic-features` 覆盖资源路径。可用 `--device cpu` 检查，但完整训练建议使用 GPU。

```text
runs/<run>/
  config.json
  train.status / test.status   # RUNNING / COMPLETED / FAILED
  checkpoints/                # 最优验证 Loss 权重及 SHA-256
  logs/train.log              # 完整训练控制台记录
  logs/test.log               # 测试记录
  results/validation.json
  results/test.json
  results/test.csv
```

分类指标在 CSV 中是0～1的比例，论文百分数需要乘100。同时输出 Has0/Non0 Acc-2、F1、Acc-5、Acc-7、MAE、Corr、Loss。测试 Loss 为样本加权主任务 MSE，不含辅助损失。请勿拿验证结果替代测试结果。

## 6. 消融及 Top-K

统一入口添加 `--variant`，仍默认同一 seed 和 batch：

| 参数 | 论文名称 | 变化 |
|---|---|---|
| full | SERA | 完整模型 |
| wo_sem | w/o SEM | 不使用语义缓存 |
| wo_ret | w/o RET | 保留全部有效描述词元，移除检索及关联辅助头 |
| wo_align | w/o ALIGN | 不执行锚点到原序列的交叉注意力 |
| wo_aux | w/o AUX | 移除辅助头及损失 |
| wo_tgf | w/o TGF | 各模态池化后拼接，保留历史 legacy_concat 实现 |

例如在第4节训练命令末尾添加 `--variant wo_align`。修改锚点预算可添加 `--top-k 4 8 16 --output runs/mosi_k4_8_16`。w/o RET 不是仅移除 Top-K 的严格单因素消融。

## 7. 可运行性检查

```bash
python tests/smoke_test.py
```

脚本创建临时的随机小型 BERT、合成原始特征及语义缓存，对 MOSI/MOSEI 分别执行数据检查、1轮训练、保存、校验和验证、重新加载和测试。结束后自动移除自建临时数据。这证明程序链路可执行，不代表真实数据上的精度或论文结果复现；真实数据检查请使用第4节命令。

## 8. 离线生成与来源

已有缓存时无需重新生成。离线工具保留在 `src/scripts/`，需在原先支持对应 Gemma 模型的专用环境中运行，其参数可查看 `--help`。当前打包不声称验证了大模型重新生成链路。原始生成提示词保存在 `src/semantic/prompts.py`；没有替换成论文后来修改的描述性模板。

本项目整理自原工程 `src`，计算模块和权重键保持兼容，主模型命名从 DashFusion 调整为 SERA。有关来源和验证范围见 `docs/IMPLEMENTATION.md`；继承代码及预训练模型的授权需在对外发布前另行核实，本包不附加未经确认的新许可证。
