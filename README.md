# SERA

## 环境安装

使用 Python 3.10，在终端执行：

```bash
git clone https://github.com/NingNing-526/SERA.git
cd SERA
conda create -n sera python=3.10 -y
conda activate sera
python -m pip install -r requirements.txt
```

GPU 运行需安装与显卡驱动兼容的 PyTorch 2.1.2。

## 运行方法

准备 BERT 权重、原始数据和语义特征，按以下结构放置：

```text
models/bert-base-uncased/           # 配置、词表和模型权重
data/mosi/unaligned_50.pkl
data/mosi/gemma_semantic_tokens_v11_len256.pt
data/mosei/unaligned_50.pkl
data/mosei/gemma_semantic_tokens_v11_len256.pt
```

在项目根目录执行 MOSI 数据检查、训练和测试：

```bash
python run.py check --dataset mosi
python run.py train --dataset mosi
python run.py test --run-dir runs/mosi_full_seed12345
```

运行 MOSEI 时，将 `mosi` 替换为 `mosei` 即可。只运行一个数据集时，只需准备对应数据。

默认配置见 `configs/default.json`，训练结果保存在 `runs/`。重新训练时添加 `--output runs/new_run`，测试时使用对应的 `--run-dir`。

自定义路径可通过 `--bert-path`、`--raw-data` 和 `--semantic-features` 指定；更多参数查看 `python run.py --help`。
