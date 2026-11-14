# 运行验证记录

验证日期：2026-11-13。主机：实验服务器（名称已匿名化），使用实验环境，CPU计算，OMP/MKL线程数为2。未占用GPU训练，未修改已有实验。

## 环境

- Python 3.8.20
- PyTorch 2.1.2+cu121
- Transformers 4.29.0
- NumPy 1.23.4
- einops 0.6.0
- pytorch-metric-learning 0.9.99
- scikit-learn 1.1.3

## 已通过

- 41个 Python 文件的 AST 语法检查及统一入口帮助输出。
- MOSI/MOSEI 两种原始特征维度与固定序列长度。
- 每个数据集：8条合成训练样本、4条验证样本、4条测试样本；batch=8，seed=12345，Top-K=8/8/16。
- 使用随机初始化的小型 BERT（1层，隐藏维768），不是预训练论文模型。
- train/valid/test 数据加载与首批前向检查。
- Full模型1轮训练，包括反向传播和优化器更新。
- 验证Loss选择与保存检查点、生成SHA-256。
- 独立测试进程读取保存配置、校验并加载权重，导出JSON和CSV；指标均为有限数值。

关键输出：

```text
PASS train: samples=8, prediction_shape=(8,)
PASS valid: samples=4, prediction_shape=(4,)
PASS test: samples=4, prediction_shape=(4,)
SMOKE PASS: mosi
PASS train: samples=8, prediction_shape=(8,)
PASS valid: samples=4, prediction_shape=(4,)
PASS test: samples=4, prediction_shape=(4,)
SMOKE PASS: mosei
```

首次尝试因服务器系统盘剩余空间不足而未能写入随机BERT权重，随后将自建临时数据转移到数据盘完成检查。测试脚本退出时已自动清理其临时数据及权重。

## 未验证

未重新执行真实数据全量训练、五个消融配置完整训练、旧权重逐位一致性验证或Gemma离线生成。README中的命令用于用户后续运行。这份记录不包含或代表论文性能结果，也不保证任意设备、依赖版本及输入文件均可运行。
