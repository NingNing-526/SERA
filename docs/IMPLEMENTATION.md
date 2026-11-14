# 实现与整理说明

## 模型路径

1. 原始模态编码：`text_encoder.py`、`audio_encoder.py`、`vision_encoder.py`。
2. 候选语义特征经 LazyLinear 和 LayerNorm 映射；`multimodal_fast_pathway.py` 完成上下文卷积、Top-K、直通门控和全部有效候选的软池化。
3. `semantic_explanation_alignment.py`：音视频锚点为 Q，原始序列为 K/V；注意力输出与锚点相加，经归一化、FFN残差归一化后，再与完整原序列沿序列维拼接。文本直接拼接。
4. `layers.py` 的 TemporalAlignment：增强文本为两路 Q，增强音视频为 K/V；两路结果相加、输出映射、文本残差、归一化、FFN残差归一化。
5. `sera.py` 注意力池化与回归；主任务 MSE 加三模态平均辅助 MSE。

默认关闭可选对齐门控、文本自注意力增强和融合模态门控，以保留当前主实验结构。保留部分未启用的内部兼容实现，不为其提供新的实验入口。

## 数据结构

原始 pickle 顶层为 train/valid/test；每个划分包含 raw_text、id、audio、vision、regression_labels，原始工程数据还包含 text。audio_lengths/vision_lengths 用于生成有效长度掩码；缺失时按完整长度处理，建议保留真实长度。

离线 torch 缓存顶层同样为 train/valid/test，每个划分包含 ids、text/audio/vision 和 text_mask/audio_mask/vision_mask。特征形状 [N,L,D]，有效性掩码 [N,L]，True 表示有效词元。ID 需要与原始划分逐项匹配，禁止跨划分重排后直接使用。

MOSI 原始音频/视觉维度为5/20，MOSEI为74/35。训练过程不读取原始媒体或在线调用大模型。

## 代码改动范围

- 主类及训练测试函数统一为 SERA、SERA_train、SERA_test，权重属性名不变。
- 替换历史环境变量配置为单一 JSON 配置和命令行入口，移除历史并行启动和结果筛选脚本。
- 统一 batch=8、seed=12345、Top-K=8/8/16；不修改原工程或历史结果。
- 测试从训练配置加载；训练拒绝覆盖已有结果；日志及状态集中保存在运行目录。
- 源自 DashFusion/TEXT/FRSR 思路的历史注释和出处保留；内部兼容代码不代表本论文提出全部基础组件。

## 验证边界

语法检查与合成数据端到端测试不等价于完整数据复现。完整 MOSI/MOSEI 训练、旧检查点逐位一致性、离线 Gemma 重新生成、GPU显存峰值及长时间稳定性需要分别验证。禁止将合成数据输出当作实验数据。
