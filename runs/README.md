# ImageNet-100 运行存档索引

Kaggle Notebooks：[主实验](https://www.kaggle.com/code/hevenshchen/intp-img-littletest)、
[XV 双 Look 共享跨度](https://www.kaggle.com/code/xiongwutao/intp-img-littletest)、
[WV 外部等变方法](https://www.kaggle.com/code/wctgy123/intp-img-littletest)

本页按消融问题组织结果，而不是按运行时间排列。除非单独注明，结果均为
ImageNet-100、`224x224` 输入、2 x T4、20 epochs。Top-5 取自最佳 Top-1
所在 epoch；运行时间是整次训练的 wall time，跨 Kaggle Version 的小幅差异
不能直接解释为结构速度差异。

## 命名

- `standard`：标准笛卡尔 Patch Embed。
- `Hex`：仅替换为 Hex patch tokenizer，不进行多姿态匹配。
- `half4d4r`：半圆内匹配 4 个方向，每个半径储存 4 个角度采样点。
- `half6d3r`：半圆内匹配 6 个方向，每个半径储存 3 个角度采样点。
- `PE`：标准位置编码。
- `Image Look`：由原图 tokenizer 响应产生的 Look Bias。
- `Center Look`：由中间 token 中心特征产生的方向 Look Bias。
- `G`：连续多少层复用一次 Center Look 探针评估；各层输出映射仍独立。
- `XV`：`xiongwutao` Kaggle Notebook 的 Version / Run 编号。

## A. 方向数与位置编码消融

这组实验比较四方向/六方向，以及 PE、Image Look 两种位置信息的组合。

| Kaggle | 几何配置 | PE | Image Look | Best Top-1 | Top-5 | 参数量 | 时间 | 本地存档目录 |
|---|---|:---:|:---:|---:|---:|---:|---:|---|
| V3-R1 | `standard` | ✓ |  | 51.52% | 79.12% | 5.544M | 152.4 min | `deit_tiny_imagenet100_ddp_e20` |
| V3-R2 | `Hex` | ✓ |  | 51.30% | 79.14% | 5.597M | 150.4 min | `deit_tiny_hex_patch_imagenet100_ddp_e20` |
| V8-R1 | `Hex + half4d4r + null-softmax` | ✓ |  | 53.56% | 80.40% | 5.486M | 152.1 min | `deit_tiny_rot_hex_harmonic_softmax_pe_imagenet100_ddp_e20` |
| V13 | `Hex + half4d4r + null-softmax` |  | ✓ | 50.26% | 77.72% | 5.463M | 163.2 min | `deit_tiny_rot_hex_harmonic_softmax_look_imagenet100_ddp_e20` |
| V12-R1 | `Hex + half4d4r + null-softmax` | ✓ | ✓ | 53.70% | 80.46% | 5.501M | 154.8 min | `deit_tiny_rot_hex_harmonic_softmax_pe_look_imagenet100_ddp_e20` |
| WV2 | `Hex + half6d3r + null-softmax` | ✓ |  | 54.54% | 80.46% | 5.464M | 151.8 min | `deit_tiny_rot_hex_harmonic_softmax_pe_half6_compact_r3_imagenet100_ddp_e20` |
| WV4-R2 | `Hex + half6d3r + null-softmax` |  | ✓ | 52.42% | 79.48% | 5.453M | 152.7 min | `deit_tiny_rot_hex_harmonic_softmax_look_half6_compact_r3_imagenet100_ddp_e20` |
| **WV4-R1** | **`Hex + half6d3r + null-softmax`** | **✓** | **✓** | **55.04%** | **81.80%** | **5.491M** | **153.2 min** | `deit_tiny_rot_hex_harmonic_softmax_pe_look_half6_compact_r3_imagenet100_ddp_e20` |

当前证据显示六方向在三种位置编码配置下均优于四方向；Image Look 单独使用
弱于 PE，但与 PE 联用时优于 PE only。

## B. Center Look 探针共享跨度消融

这组固定为 `Hex + half6d3r + null-softmax + PE + Center Look`，只改变
`G`。Center Look 实际作用于前 11 个 Transformer blocks，因此最后一组可以
不足 `G` 层。

| Kaggle | G | 探针组数 | Best Top-1 | Top-5 | 参数量 | 时间 | 本地存档目录 |
|---|---:|---:|---:|---:|---:|---:|---|
| V19-R1 | 1 | 11 | 54.64% | 81.86% | 5.478M | 146.7 min | `deit_tiny_rot_hex_harmonic_softmax_pe_center_grid_look_share1l_half6_compact_r3_imagenet100_ddp_e20` |
| V19-R2 | 2 | 6 | 54.64% | 81.56% | 5.472M | 146.7 min | `deit_tiny_rot_hex_harmonic_softmax_pe_center_grid_look_share2l_half6_compact_r3_imagenet100_ddp_e20` |
| V20-R1 | 3 | 4 | 55.00% | 81.56% | 5.470M | 155.8 min | `deit_tiny_rot_hex_harmonic_softmax_pe_center_grid_look_share3l_half6_compact_r3_imagenet100_ddp_e20` |
| **V20-R2** | **4** | **3** | **55.18%** | **81.66%** | **5.469M** | **155.1 min** | `deit_tiny_rot_hex_harmonic_softmax_pe_center_grid_look_share4l_half6_compact_r3_imagenet100_ddp_e20` |
| V20-R3 | 6 | 2 | 54.46% | 81.60% | 5.467M | 155.0 min | `deit_tiny_rot_hex_harmonic_softmax_pe_center_grid_look_share6l_half6_compact_r3_imagenet100_ddp_e20` |
| V22 | 12 | 1 | 54.48% | 81.38% | 5.466M | 145.2 min | `deit_tiny_rot_hex_harmonic_softmax_pe_center_grid_look_share12l_half6_compact_r3_imagenet100_ddp_e20` |

在 PE + Center Look 单支路中，G4 当前最好。G6 与 G12 都回落到约 54.5%，
说明探针适度跨层共享有效，但全局化后会损失层级差异；这一最优跨度不直接
适用于同时启用 Image Look 的组合。

## C. Center Look 放置组合与共享跨度交互

早期放置消融把 G4 当作固定设置，因此曾得到“双 Look 没有额外收益”的结论。
后续共享跨度扫描表明，放置方式与 `G` 存在明显交互：Center Look 单独配合 PE
时 G4 最好，而同时启用 Image Look 时 G3 最好。因而这里分别呈现当前关键直接
比较和历史 G4 完整对照，不再把某一个跨度的结果推广为两种 Look 的一般关系。

### C1. 当前 G3 关键直接比较

| 配置 | PE | Image Look | Center Look | Best Top-1 | Top-5 | 参数量 | 时间 | 状态/存档目录 |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---|
| PE only | ✓ |  |  | 54.54% | 80.46% | 5.464M | 151.8 min | `deit_tiny_rot_hex_harmonic_softmax_pe_half6_compact_r3_imagenet100_ddp_e20` |
| PE + Image Look | ✓ | ✓ |  | 55.04% | 81.80% | 5.491M | 153.2 min | `deit_tiny_rot_hex_harmonic_softmax_pe_look_half6_compact_r3_imagenet100_ddp_e20` |
| PE + G3 Center Look | ✓ |  | G3 | 55.00% | 81.56% | 5.470M | 155.8 min | `deit_tiny_rot_hex_harmonic_softmax_pe_center_grid_look_share3l_half6_compact_r3_imagenet100_ddp_e20` |
| **PE + Image Look + G3 Center Look** | **✓** | **✓** | **G3** | **55.54%** | **81.82%** | **5.497M** | **172.0 min** | `deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share3l_half6_compact_r3_optimized_imagenet100_ddp_e20` |

在相同 G3 下，双 Look 比 PE + Center Look 高 0.54 个 Top-1 百分点；相对于
PE + Image Look 也提高 0.50 点。这说明 Image Look 与 Center Look 可以互补，
但互补性依赖合适的跨层复用跨度，而不是无条件成立。

### C2. 历史 G4 完整放置对照

| 配置 | PE | Image Look | Center Look | Best Top-1 | Top-5 | 参数量 | 时间 | 状态/存档目录 |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---|
| PE only | ✓ |  |  | 54.54% | 80.46% | 5.464M | 151.8 min | `deit_tiny_rot_hex_harmonic_softmax_pe_half6_compact_r3_imagenet100_ddp_e20` |
| Image Look only |  | ✓ |  | 52.42% | 79.48% | 5.453M | 152.7 min | `deit_tiny_rot_hex_harmonic_softmax_look_half6_compact_r3_imagenet100_ddp_e20` |
| PE + Image Look | ✓ | ✓ |  | 55.04% | 81.80% | 5.491M | 153.2 min | `deit_tiny_rot_hex_harmonic_softmax_pe_look_half6_compact_r3_imagenet100_ddp_e20` |
| PE + G4 Center Look | ✓ |  | G4 | 55.18% | 81.66% | 5.469M | 155.1 min | `deit_tiny_rot_hex_harmonic_softmax_pe_center_grid_look_share4l_half6_compact_r3_imagenet100_ddp_e20` |
| G4 Center Look only (V24) |  |  | G4 | 52.34% | 79.22% | 5.431M | 165.5 min | `deit_tiny_rot_hex_harmonic_softmax_center_grid_look_only_share4l_half6_compact_r3_imagenet100_ddp_e20` |
| PE + Image Look + G4 Center Look (V23) | ✓ | ✓ | G4 | 54.96% | 81.84% | 5.496M | 207.4 min | `deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share4l_half6_compact_r3_imagenet100_ddp_e20` |

G4 下，三路组合的 54.96% 确实未超过两个单 Look 分支；但结合 C1 与 D 节，
应将其解释为共享跨度与放置组合不匹配，而不是两种 Look 在机制上互相冲突。
V23 仍是优化前的历史运行；若论文需要严格比较 G3/G4 的双 Look 曲线，应再用
优化后实现补跑 G4。所有差异目前仍来自单随机种子，正式统计需要均值和方差。

## D. 双 Look 下的 Center Look 共享跨度

这组固定使用 `Hex + half6d3r + null-softmax + PE + Image Look + Center Look`，
只改变 `G`。除带 `*` 的 G4 外，均使用提交 `6f06787` 中优化后的结构化
Look 内核；五组优化运行来自 XV Notebook，训练配置、随机种子和预算对齐。

| Kaggle | G | 探针组数 | Best Top-1 | Top-5 | 参数量 | 时间 | 本地存档目录 |
|---|---:|---:|---:|---:|---:|---:|---|
| XV1-R1 | 1 | 11 | 54.76% | 82.20% | 5.505M | 171.5 min | `deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share1l_half6_compact_r3_optimized_imagenet100_ddp_e20` |
| XV1-R2 | 2 | 6 | 55.02% | 81.84% | 5.499M | 169.8 min | `deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share2l_half6_compact_r3_optimized_imagenet100_ddp_e20` |
| **XV2-R1** | **3** | **4** | **55.54%** | **81.82%** | **5.497M** | **172.0 min** | `deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share3l_half6_compact_r3_optimized_imagenet100_ddp_e20` |
| V23* | 4 | 3 | 54.96% | 81.84% | 5.496M | 207.4 min | `deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share4l_half6_compact_r3_imagenet100_ddp_e20` |
| XV2-R2 | 6 | 2 | 54.92% | 81.82% | 5.495M | 170.8 min | `deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share6l_half6_compact_r3_optimized_imagenet100_ddp_e20` |
| XV2-R3 | 12 | 1 | 54.70% | 81.82% | 5.493M | 171.0 min | `deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share12l_half6_compact_r3_optimized_imagenet100_ddp_e20` |

G3 当前最好：比 PE + Image Look 高 0.50 个百分点，比相同 G3 的
PE + Center Look 高 0.54 个百分点。G1 探测过密、G6/G12 共享过宽时均有
回落，说明双 Look 的互补性依赖中等跨度的 Feature Look 更新。五组优化运行
稳定在 169.8--172.0 分钟；V23* 使用优化前的静态展开内核，因此其 207.4
分钟只用于记录历史，不能作为 G4 本身的结构耗时。

## E. 文献中的旋转/等变 ViT 对照

这组三项不是 SHARE 自身消融，而是依据已发表方法构建的外部结构对照。训练
数据、输入尺寸、20-epoch 预算、优化器日程和有效 global batch 均与 Standard
对齐；但它们是面向当前 ImageNet-100/DeiT-Tiny 口径的算子级复现，而不是对
原论文数据集、完整网络与训练 recipe 的逐项复刻。

| 方法 | Kaggle Notebook | 改造层级 | 核心机制 | Best Top-1 | Top-5 | 相对 Standard Top-1 | 参数量 | 时间 |
|---|---|---|---|---:|---:|---:|---:|---:|
| Equi-ViT / GMR + PE | [hevenshchen · V346414357](https://www.kaggle.com/code/hevenshchen/intp-img-littletest?scriptVersionId=346414357) | Tokenizer | 两层 GMR 环形/高斯核，固定几何等变滤波器组 | 41.54% | 70.42% | -9.98 | 5.424M | 151.0 min |
| ARC Adaptive + PE | [hevenshchen · V346414357](https://www.kaggle.com/code/hevenshchen/intp-img-littletest?scriptVersionId=346414357) | Tokenizer | 输入条件路由预测核混合权重与连续角度，再旋转并合成卷积核 | 49.38% | 76.56% | -2.14 | 5.986M | 155.2 min |
| GE-ViT p4 local | [wctgy123 · WV18](https://www.kaggle.com/code/wctgy123/intp-img-littletest?scriptVersionId=346502718) | 完整骨干 | C4 方向特征场、局部群自注意力与方向作用的相对位置 | **56.06%** | **82.06%** | **+4.54** | 5.509M | 166.8 min |

### Equi-ViT / GMR

- 保留标准 `14x14` 笛卡尔 token 网格、DeiT-Tiny Transformer 与可学习 PE。
- Tokenizer 采用 Equi-ViT 的 GMR ring/Gaussian 参数化以及 `6x6 -> 11x11`
  两级顺序滤波器组；本项目以 stride `6/2`、中间宽度 24 将其适配为
  `224 -> 37 -> 14`，使输出 token 数和模型规模接近 Standard。
- 该结果说明固定的旋转等变核并不会自动适应当前训练口径：参数量和耗时均未
  增加，但 Top-1 明显回落。因此它适合作为“预设等变结构”的负向对照，而
  不能据此否定原论文在其原始任务与训练 recipe 下的结论。

### ARC Adaptive

- 同样只替换 DeiT 的 `16x16` patch projection，其余骨干和 PE 保持一致。
- 使用四个 canonical patch kernels；路由器依据输入预测 sigmoid 混合权重和
  softsign 有界角度（`+/-40` 度），各核经双线性旋转后先加权合成，再只执行
  一次卷积。实现中的 batch chunking 只约束临时旋转核显存，不改变计算定义。
- ARC 比 GMR 高 7.84 个 Top-1 百分点，但仍比 Standard 低 2.14 点，构成
  “先预测姿态、再定向旋转”的自适应对照；它与 SHARE 广搜离散姿态并通过
  null-softmax 路由的机制不同。

### GE-ViT p4 local

- GE-ViT 的方向轴贯穿骨干，因而不能只把其 tokenizer 接到普通 ViT 上。本实现
  保留联合 Cartesian x C4 特征场、空间与方向邻域上的局部群自注意力、由查询
  方向作用于相对坐标的位置项，以及 spatial-sum/orientation-max 分类读出。
- 使用共享 C4 lifting patch bank、`14x14 -> 7x7 -> 3x3` 的 `2/2/2` pooled
  stages、宽度 `144/288/336`、每层 3 heads 和 `5x5` 局部窗口；参数量因此与
  DeiT-Tiny 对齐。由于每个 token 保留四个方向状态，训练采用每卡 microbatch
  128、两步梯度累计，仍保持有效 global batch 512。
- WV18 在第 19 轮达到最佳 `56.06/82.06`，第 20 轮为 `55.88/81.98`。
  它是当前表中最强的外部等变基线，比最强 SHARE 双 Look 结果
  `55.54/81.82` 高 0.52/0.24 点；但两者改变的网络范围不同，论文中应将其作为
  强结构基线，而非同一插件的直接消融。

GMR 与 ARC 在 `hevenshchen` Notebook 的同一个 Script Version `346414357`
中顺序运行；GE-ViT 来自 `wctgy123` Notebook 的 WV18（Script Version
`346502718`）。三项均为单随机种子结果，现阶段只支持在当前固定预算下比较，
后续正式统计仍需补充多随机种子均值和方差。

## 弃用或非对齐实验

| Kaggle | 配置 | Best Top-1 | 原因 | 存档目录 |
|---|---|---:|---|---|
| V7 | `Hex + half4d4r + PE`，Tokenizer 无 null-softmax | 50.94% | 旧 harmonic 聚合，不属于当前对齐主线 | `deit_tiny_rot_hex_harmonic_pe_imagenet100_ddp_e20` |
| V8-R2 | 分组三段 V + K12 尺度补偿 | 42.96% | 聚合结构明显较差 | `deit_tiny_rot_hex_dot_grouped_compensated_pe_imagenet100_ddp_e20` |
| V10 | 负 L1 距离核 | 36.12% | 收敛和最终性能均较差 | `deit_tiny_rot_hex_harmonic_l1_softmax_pe_imagenet100_ddp_e20` |

## 存档约定

- `summary.json` 保存最终摘要和逐 epoch 历史，`metrics.jsonl` 保存流式训练记录。
- `model_summary.json` 保存参数量和结构诊断；完整存档可能另含 `best.pt`、`last.pt`。
- 主表只纳入 `status = complete` 且训练预算对齐的结果。
- Kaggle `Version` 是 Notebook 保存版本；`R1/R2/...` 表示同一 Version 内的执行顺序。
- 新实验完成后，优先从真实 `summary.json` 更新本页，禁止按日志截图或文件名猜指标。
