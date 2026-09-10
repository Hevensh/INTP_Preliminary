# ImageNet-100 server50 结果核对

来源：本地原始 results_no_weights.zip（原 ZIP 不上传；校验值见 provenance.json）。仅统计 11 个 complete 的 DALI run；另外 11 个旧任务均无完成轮次，不参与比较。

## 口径

seed 2027；50 epochs；224；batch 512；单卡；DALI 1.50.0；AdamW lr=0.0005，5 epoch warmup 后恒定。每轮训练 129536 张（drop-last），验证 5000 张。八个 MAMS 变体均为 K24 单尺度、r3、PE only，无 Look 和分化。

Top-5 取最佳 Top-1 对应轮次；时间取该次运行 wall time，含训练及验证，非纯算子性能。

| 配置 | 最佳 Top-1 % | 轮次 | 对应 Top-5 % | 最终 Top-1 % | 全程分钟 |
|---|---:|---:|---:|---:|---:|
| ViT | 64.70 | 50 | 86.92 | 64.70 | 89.9 |
| Equi/GMR 本地适配 | 58.26 | 50 | 83.28 | 58.26 | 90.7 |
| ARC 本地适配 | 64.18 | 50 | 86.04 | 64.18 | 225.6 |
| Square half4d3r | 65.08 | 50 | 86.44 | 65.08 | 89.9 |
| Square half6d3r | 65.70 | 48 | 87.04 | 65.54 | 89.8 |
| Square full4d3r | 64.66 | 50 | 86.64 | 64.66 | 90.0 |
| Square full6d3r | 64.46 | 49 | 86.40 | 64.18 | 90.4 |
| Hex half4d3r | 65.20 | 50 | 87.30 | 65.20 | 89.8 |
| Hex half6d3r | 64.44 | 48 | 86.56 | 64.18 | 92.3 |
| Hex full4d3r | 65.08 | 48 | 87.46 | 64.74 | 90.6 |
| Hex full6d3r | 63.76 | 44 | 86.16 | 63.72 | 93.3 |

## 解释

- Square half6d3r 比 ViT 最佳值高 1.00 个百分点，最终轮高 0.84 个百分点。最后十轮平均为 64.81%，ViT 为 63.79%，不只是挑一个峰值才领先。
- 半角度前期更快：第 20 轮四个半角度版本为 53.36–54.30%，四个全角度版本为 50.56–51.68%，ViT 为 50.88%。不能仅凭此归因于一阶矩抵消，仍需响应分布证据。
- Hex 不全面占优：half6 下比 Square 低 1.26 个百分点；half4/full4 下分别高 0.12/0.42 个百分点。Hex 的独立收益尚未确立。
- MAMS 参数约 5.464M，ViT 5.544M；全程耗时约 90 分钟，端到端已接近。并发运行和输入流水线会掩盖计算差异，不代表 FLOPs 或推理速度相等。
- 多个模型第 50 轮仍创最好值，不能称充分收敛。若延长训练，应对 ViT、Square half6d3r、Hex half4d3r 等量延长。

## 核验与局限

ZIP CRC 通过；11 组逐轮记录均恰为 1–50；metrics.jsonl 与 summary.history 完全一致；收集表最高值与逐轮重算一致；公共训练配置及样本数一致。

仅一个种子，未经显著性检验；验证集也用于挑选最佳 epoch。不能和旧 CPU/cosine/20 epoch 实验直接归因比较。没有权重或逐样本预测，未复算模型预测。


## 目录与复核

- `collected_summary.json`：11 个完成实验的收集汇总。
- 各运行目录 `metrics.jsonl`：50 轮训练/验证指标，数值原样保留。
- `config.json`：实际运行配置；仅 data_root/output_root 改为占位符，使用时须替换。
- `model_summary.json`：原始参数统计；文本换行可能规范化。
- `runtime.json`：Python、CUDA、PyTorch、DALI、GPU 和 batch 环境，去除服务器路径。
- `provenance.json`：原 ZIP SHA256、ZIP 内源文件字节校验值和清理说明。

数据源：[ambityga/imagenet100](https://www.kaggle.com/datasets/ambityga/imagenet100)，此次使用版本 8。数据集本身遵守上游许可，不随代码仓库重新分发。

此提交是结果存档，不是服务器训练代码的完整快照。原环境未记录 Git commit；不能把此结果提交号当作训练源代码版本。完整原 ZIP 留在本地 `runs/runs/`，数据集原图、权重、缓存与个人路径不提交。
