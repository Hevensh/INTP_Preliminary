# ImageNet-100 运行存档索引

Kaggle Notebooks：[主实验](https://www.kaggle.com/code/hevenshchen/intp-img-littletest)、
[XV 双 Look 共享跨度](https://www.kaggle.com/code/xiongwutao/intp-img-littletest)

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

G4 当前最好。G6 与 G12 都回落到约 54.5%，说明探针适度跨层共享有效，
但全局化后会损失层级差异。

## C. Center Look 放置组合消融

这组固定使用当前最优的 `half6d3r + G4`，用于区分 PE、Image Look 和
Center Look 的独立贡献。六项均已完成 20 轮对齐训练。

| 配置 | PE | Image Look | G4 Center Look | Best Top-1 | Top-5 | 参数量 | 时间 | 状态/存档目录 |
|---|:---:|:---:|:---:|---:|---:|---:|---:|---|
| PE only | ✓ |  |  | 54.54% | 80.46% | 5.464M | 151.8 min | `deit_tiny_rot_hex_harmonic_softmax_pe_half6_compact_r3_imagenet100_ddp_e20` |
| Image Look only |  | ✓ |  | 52.42% | 79.48% | 5.453M | 152.7 min | `deit_tiny_rot_hex_harmonic_softmax_look_half6_compact_r3_imagenet100_ddp_e20` |
| PE + Image Look | ✓ | ✓ |  | 55.04% | 81.80% | 5.491M | 153.2 min | `deit_tiny_rot_hex_harmonic_softmax_pe_look_half6_compact_r3_imagenet100_ddp_e20` |
| PE + G4 Center Look | ✓ |  | ✓ | 55.18% | 81.66% | 5.469M | 155.1 min | `deit_tiny_rot_hex_harmonic_softmax_pe_center_grid_look_share4l_half6_compact_r3_imagenet100_ddp_e20` |
| G4 Center Look only (V24) |  |  | ✓ | 52.34% | 79.22% | 5.431M | 165.5 min | `deit_tiny_rot_hex_harmonic_softmax_center_grid_look_only_share4l_half6_compact_r3_imagenet100_ddp_e20` |
| PE + Image Look + G4 Center Look (V23) | ✓ | ✓ | ✓ | 54.96% | 81.84% | 5.496M | 207.4 min | `deit_tiny_rot_hex_harmonic_softmax_pe_look_center_grid_look_share4l_half6_compact_r3_imagenet100_ddp_e20` |

Center Look 单独使用达到 52.34%，与 Image Look only 的 52.42% 接近，但明显
弱于 PE。三路同时加入达到 54.96%，没有超过 PE + Image Look 的 55.04% 或
PE + G4 Center Look 的 55.18%。当前单次训练证据因此更支持把 Image Look 与
Center Look 视为可替换/可组合的消融分支，而不是默认同时启用；差异小于一个
百分点，仍需多随机种子验证。

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
