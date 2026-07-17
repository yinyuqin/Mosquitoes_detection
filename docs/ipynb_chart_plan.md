# Notebook 图表生成规划（基于最终 7 页 PPT）

## 1. 来源记录

- 来源文件：`D:/浏览器下载/PPT的图片版.zip`
- SHA256：`71E4691F5D6378F2ABE12AB5F0AB2615CB59E965E6E2625252CB60B8A89286C1`
- 记录日期：2026-07-17
- ZIP 内目录：`mosquito_ppt_pages_02-07/`
- ZIP 内含 7 张页面图片和 1 个仅含封面的 PowerPoint 文件。

| 文件 | 尺寸 | 作用 |
|---|---:|---|
| `01-cover.png` | 1672×941 | 封面，不需要 Notebook 复现 |
| `01-cover.pptx` | - | 封面 PowerPoint，不作为 Notebook 输入 |
| `02_data_and_mfcc.png` | 1920×1080 | 数据构成、OOD 构成与 MFCC 参数 |
| `03_windowing_and_topk.png` | 1920×1080 | 短样本补零、长样本切窗与 Top-3 Mean |
| `04_baseline_evolution.png` | 1920×1080 | SVM Baseline 到正式 MIL 的方法演进 |
| `05_model_matrix_and_protocol.png` | 1920×1080 | Global/MIL × Logistic/MLP 模型矩阵与评估协议 |
| `06_id_ood_results.png` | 1920×1080 | ID/OOD 多模型成绩表与三条原因分析 |
| `07_conclusion.png` | 1672×941 | 最终模型、主要结论、局限与下一步 |

本文件只记录 PPT 中与 Notebook 可复现性有关的内容。原始 PPT 图片不复制进项目，后续以外部 ZIP 为准。

## 2. 对 7 页 PPT 的理解

### 第 1 页：封面

核心信息：项目使用 MFCC 与窗口化 MIL 检测蚊子声音，重点不是 ID 最高分，而是真实 OOD 泛化。

Notebook 责任：无需复现封面插画；只需在 Notebook 标题和研究目标中保持同一研究问题。

### 第 2 页：数据与 MFCC

PPT 展示：

- HumBugDB 有 9,115 个有效片段；
- 6,683 个蚊子、2,432 个背景，蚊子占 73.3%；
- OOD 背景共 170 个：`data1=70`、`data2=100`；
- `data2` 的 100 个片段来自同一条 `origin.wav`，不代表 100 个独立环境；
- OOD 蚊子共 1,699 个：Vasconcelos 1,689、other 10；
- 另有 8 个零长度文件被排除；
- MFCC 参数：8 kHz、FFT 512、hop 256、13 MFCC，加一阶和二阶 Delta 后为 39 通道；
- 保存 `labels.npy`、`groups.npy`、`sample_ids.npy`；
- 使用 group-disjoint split。

Notebook 责任：数据数量必须从实际文件/manifest 计算，不能手写；生成数据构成图、来源表和至少一个真实 MFCC 示例。

### 第 3 页：窗口化与 Top-3 Mean

PPT 展示：

- 短样本训练时随机左右补零到 64 帧；
- 验证/测试使用居中补零；
- 长样本使用 64 帧窗口、32 帧步长、50% overlap；
- 末尾不能完整覆盖时增加右对齐窗口；
- 窗口概率取最高 3 个的平均值；
- Top-3 Mean 比 Max Pooling 更不易被单个异常窗口触发。

Notebook 责任：用真正的窗口构造函数生成图，而不是绘制与代码无关的示意图；短样本、长样本和 Top-3 聚合必须共享模型实际使用的函数。

### 第 4 页：Baseline 演进

PPT 展示：

1. SVM Baseline：整段 39 通道 Mean+Std，形成 78 维特征；
2. SVM Pro：窗口 SVM + Max Pooling，FPR 高；
3. SVM Pro2：改为 Top-K Mean，误报改善但窗口标签仍较粗糙；
4. SVM Large：约 2,432 维输入，训练成本过高；
5. 最终得到三个设计原则：局部窗口有效、Max Pooling 过敏、需要片段标签训练的 MIL。

Notebook 责任：该页是方法历史，不需要重新训练 SVM，也不应该编造旧 SVM 数字。Notebook 用一张 Markdown/DataFrame 路线表记录方法、聚合方式和观察即可。

### 第 5 页：正式模型矩阵与协议

PPT 展示：

- 两个实验问题：窗口化是否有效、非线性是否有效；
- 四类模型：Logistic Global、Logistic MIL、MLP Global、MLP MIL；
- MLP 结构为 `78→128→64→1`，包含 LayerNorm、GELU、Dropout(0.2)，参数量 18,817；
- 20% 原始录音组作为测试集；
- 其余数据进行 5-fold Stratified Group CV；
- 每折独立拟合标准化参数；
- OOF 预测选择 F1 最优阈值；
- 5 个 fold 测试概率取平均；
- 旧 Global 模型采用 sample-level split，仅作探索性参考。

Notebook 责任：完整训练代码只保留最终 Logistic MIL。模型矩阵可以由元数据表生成；其他模型不放训练代码，只在结果复现节读取冻结预测或现有 metrics 文件。

### 第 6 页：ID/OOD 结果

PPT 展示的结果表：

| 模型 | 切分 | ID F1 | ID FPR | OOD Accuracy | OOD Recall | OOD FPR |
|---|---|---:|---:|---:|---:|---:|
| Logistic Global* | sample | 95.86% | 9.26% | 89.12% | 95.29% | 17.06% |
| Logistic MIL | grouped | 93.50% | 8.13% | 94.41% | 100.00% | 11.18% |
| MLP Global* | sample | 98.84% | 3.91% | 78.53% | 100.00% | 42.94% |
| MLP Global Grouped | grouped | 99.00% | 2.44% | 79.12% | 100.00% | 41.76% |
| MLP MIL | grouped | 95.95% | 9.55% | 91.18% | 100.00% | 17.65% |

三条分析：

1. `MLP Global Grouped → MLP MIL`：OOD Accuracy 由 79.12% 升至 91.18%，OOD FPR 由 41.76% 降至 17.65%，说明窗口化有效；
2. MLP Global Grouped 的 OOD ROC-AUC 为 99.40%，但固定阈值 FPR 为 41.76%，说明主要问题是阈值/概率迁移，不是完全失去排序能力；
3. Logistic MIL 达到 OOD Recall 100%、OOD FPR 11.18%、OOD Balanced Accuracy 94.41%，是当前最佳权衡。

Notebook 责任：这是最严格的复现项。表格必须由保存的标签、概率、阈值和统一评价函数重新计算，不能把 PPT 数字直接写入 DataFrame。

### 第 7 页：结论

PPT 展示：

- MFCC Baseline 有效；
- 窗口化提升 OOD 稳定性；
- 模型容量不等于真实泛化；
- 最终模型为 Logistic MIL；
- 局限：OOD 背景仅 170 个、data2 同源、11.18% FPR 仍偏高；
- 下一步：更多独立录音、Threshold-Recall-FPR 曲线、面向目标 FPR 的部署阈值。

Notebook 责任：最终模型卡片中的三个指标必须由 OOD 评价字典自动提取。局限由 OOD manifest 自动生成，不能与实际数据数量脱节。

## 3. Notebook 总体结构

Notebook 类型：`experiment`。

建议输出文件：

```text
output/jupyter-notebook/mosquito_logistic_mil.ipynb
```

建议章节：

1. **Title, objective and reproducibility contract**
2. **Imports, paths, style and random seed**
3. **Dataset manifest and data-source documentation**
4. **Audio examples and MFCC extraction**
5. **Group-disjoint train/test split**
6. **Short-sample padding and long-sample windowing**
7. **Top-3 Mean pooling demonstration**
8. **Final Logistic MIL model and loss**
9. **Five-fold training or checkpoint loading**
10. **OOF threshold selection**
11. **ID evaluation**
12. **OOD manifest, full set and balanced selection**
13. **OOD evaluation with the frozen threshold**
14. **PPT comparison-table reproduction from frozen predictions**
15. **Conclusions, limitations and artifact index**

Notebook 默认应能快速执行评估；完整训练用显式开关控制：

```python
SEED = 2026
RUN_FULL_PREPROCESS = False
RUN_TRAINING = False
REBUILD_ALL_FIGURES = True
```

`RUN_TRAINING=False` 时读取现有五折 Logistic MIL checkpoint；`RUN_TRAINING=True` 时完整训练最终模型。其他模型没有训练开关，只读取冻结预测以复现第 6 页成绩表。

## 4. PPT 到 Notebook 图表映射

### 图表 A：数据集与 OOD 构成

- 对应 PPT：第 2 页。
- Notebook 输入：
  - `data/processed/humbugdb_mfcc/labels.npy`
  - `data/processed/humbugdb_mfcc/groups.npy`
  - OOD ZIP 文件扫描结果或 `outputs/ood_evaluation/metrics.json` 的 dataset manifest
- 生成内容：
  - HumBugDB 蚊子/背景 donut chart；
  - OOD background 来源条形图（70/100）；
  - OOD mosquito 来源条形图（1689/10）；
  - 数据来源与独立性表格。
- 输出：
  - `outputs/notebook_figures/fig_02_dataset_composition.png`
  - `outputs/notebook_tables/table_02_dataset_manifest.csv`
- 断言：

```python
assert n_humbug == 9115
assert n_id_positive == 6683
assert n_id_negative == 2432
assert n_ood_negative == 170
assert n_ood_positive == 1699
assert n_failed_ood == 8
```

### 图表 B：真实 MFCC 示例

- 对应 PPT：第 2 页 MFCC 处理流程的实证补充。
- Notebook 输入：一条 HumBugDB 蚊子音频、一条 HumBugDB 背景、一条 OOD 蚊子、一条 OOD 背景。
- 生成内容：每条样本显示 waveform、MFCC、Delta、Delta²，并提供 `IPython.display.Audio`。
- 输出：
  - `outputs/notebook_figures/fig_02_mfcc_examples.png`
  - `outputs/notebook_tables/table_02_audio_examples.csv`
- 要求：图片标题包含数据来源、真实标签、时长和采样率。

### 图表 C：短样本补零

- 对应 PPT：第 3 页左侧。
- Notebook 输入：真实有效帧少于 64 的样本；若完整数据中没有合适样本，则从真实特征裁切出确定性演示片段，并明确标注为 demonstration。
- 生成内容：同一特征在 3 个固定 seed 下的左/中/右补零位置；另显示验证/测试的居中补零。
- 输出：`outputs/notebook_figures/fig_03_short_padding.png`
- 实现约束：绘图必须调用训练实际使用的 padding 函数。

### 图表 D：长样本窗口切分

- 对应 PPT：第 3 页右侧。
- Notebook 输入：一条帧长明显大于 64 的真实样本。
- 生成内容：在 MFCC 时间轴上覆盖窗口矩形，标出窗口长 64、步长 32、50% overlap 和最后的右对齐窗口。
- 输出：`outputs/notebook_figures/fig_03_long_windowing.png`
- 断言：最后一个窗口必须覆盖最后一个有效帧。

### 图表 E：Top-3 Mean 聚合

- 对应 PPT：第 3 页下方。
- Notebook 输入：最终 Logistic MIL 对一条真实片段的逐窗口概率。
- 生成内容：所有窗口概率柱状图，最高 3 个使用深蓝色；标注 Top-3 Mean 和最终 clip probability。
- 输出：`outputs/notebook_figures/fig_03_top3_pooling.png`
- 断言：图中 clip probability 与模型实际聚合结果数值一致。

### 表格 F：Baseline 演进记录

- 对应 PPT：第 4 页。
- Notebook 输入：方法元数据，不读取或伪造缺失的 SVM 数字。
- 生成内容：`Method / Feature unit / Aggregation / Observation / Formal candidate` 五列 DataFrame。
- 输出：`outputs/notebook_tables/table_04_baseline_evolution.csv`
- 说明：该页属于方法路线，不需要生成结果柱状图。

### 表格 G：模型矩阵与协议

- 对应 PPT：第 5 页。
- Notebook 输入：模型元数据和最终 Logistic MIL 配置。
- 生成内容：Global/MIL × Logistic/MLP 矩阵；训练协议表；MLP 参数量从模型定义或 checkpoint metadata 读取。
- 输出：
  - `outputs/notebook_tables/table_05_model_matrix.csv`
  - `outputs/notebook_tables/table_05_protocol.csv`
- 说明：Notebook 不包含其他模型训练代码；表格仅用于解释实验设计。

### 表格 H：ID/OOD 多模型成绩表

- 对应 PPT：第 6 页，是最重要的复现产物。
- Notebook 输入：
  - 各模型 `test_indices.npy`
  - 各模型 `test_probabilities.npy`
  - `data/processed/humbugdb_mfcc/labels.npy`
  - 各模型训练 OOF 阈值
  - `outputs/ood_evaluation_balanced/probabilities.npz`
  - OOD 平衡集标签与选择索引
- 计算方式：统一调用一个 `compute_binary_metrics(labels, probabilities, threshold)` 函数，分别计算 ID F1、ID FPR、OOD Accuracy、OOD Recall、OOD FPR 和 OOD ROC-AUC。
- 输出：
  - `outputs/notebook_tables/table_06_model_comparison.csv`
  - `outputs/notebook_figures/table_06_model_comparison.png`
- 验证：重新计算结果必须与各模型 `metrics.json` 在 `1e-6` 容差内一致。
- 展示规则：sample-level 模型带 `*`，不得把其 ID 结果与 grouped 模型称为完全公平比较。

### 图表 I：ID/OOD 差距与窗口化收益

- 对应 PPT：第 6 页下方三条分析。
- Notebook 输入：表格 H 的结果。
- 生成内容：
  - `MLP Global Grouped → MLP MIL` 的 OOD Accuracy slope chart；
  - 同一比较的 OOD FPR slope chart；
  - Logistic MIL、MLP Global Grouped、MLP MIL 的 OOD Recall/FPR 对照散点图。
- 输出：`outputs/notebook_figures/fig_06_ood_tradeoff.png`
- 备注：PPT 当前以分析卡片表现这些数字；Notebook 用数据图提供可复现证据。

### 图表 J：MLP 阈值迁移证据

- 对应 PPT：第 6 页“MLP 的问题是阈值迁移”。
- Notebook 输入：MLP Global Grouped 的 ID/OOD 预测概率与固定阈值 0.20。
- 生成内容：ID 与 OOD 背景概率分布叠加图，画出固定阈值；同时在图角标注 OOD ROC-AUC 99.40% 和固定阈值 FPR 41.76%。
- 输出：`outputs/notebook_figures/fig_06_mlp_threshold_shift.png`
- 解释边界：该图支持“概率/阈值迁移”，不应写成“MLP 完全没有 OOD 排序能力”。

### 表格 K：最终 Logistic MIL 模型卡

- 对应 PPT：第 7 页。
- Notebook 输入：平衡 OOD Logistic MIL 评价字典。
- 生成内容：自动提取 OOD Recall、FPR、Balanced Accuracy，同时附上阈值来源和样本数量。
- 输出：`outputs/notebook_tables/table_07_final_model_card.csv`
- 断言：

```python
assert recall == 1.0
assert round(fpr * 100, 2) == 11.18
assert round(balanced_accuracy * 100, 2) == 94.41
```

### 可选图表 L：Threshold-Recall-FPR 曲线

- 对应 PPT：第 7 页“下一步”。
- Notebook 输入：训练 OOF 概率，不使用 OOD 调阈值。
- 生成内容：threshold、Recall、FPR、F1 四者关系，标出当前 F1 最优阈值 0.32，并报告 FPR 目标工作点。
- 输出：`outputs/notebook_figures/fig_07_threshold_recall_fpr.png`
- 规则：若将该图作为正式结果，应同步把 PPT 中的“下一步绘制”改为“部署分析”；否则只放 Notebook appendix。

## 5. 统一图表风格

Notebook 图表应与 PPT 视觉相容，但不需要复刻 PPT 装饰边框。

```python
PPT_NAVY = "#1E2C57"
PPT_BLUE = "#536AA3"
PPT_IVORY = "#F5F1E8"
PPT_RED = "#9F3944"
PPT_GREEN = "#477565"
PPT_GRAY = "#A3A7B2"
```

要求：

- 白底或浅米白底；
- 主要模型/正式结果用 navy；
- 高 FPR 警告用 red；
- 最终选择用 green；
- 图片保存为 PNG 300 dpi，同时保留 SVG/PDF 矢量版本；
- 数字统一显示两位百分比；
- 所有轴、图例和标题使用可嵌入的中英文字体；
- 不依赖 Notebook 隐藏状态，所有图可从上到下重新生成。

## 6. 复现与防泄漏检查

Notebook 在生成最终表格前必须执行：

1. 检查 HumBugDB train/test group 交集为 0；
2. 检查 grouped 三个正式模型的测试索引一致；
3. 检查标准化参数来自每折训练部分；
4. 检查阈值来自训练 OOF，而不是 ID test 或 OOD；
5. 检查 OOD 平衡选择使用 seed 2026，保留全部 170 个负样本并选择 170 个正样本；
6. 检查 `data2` 独立性说明出现在 manifest 和结论中；
7. 检查所有 PPT 数字均由代码产生并与 metrics 文件一致；
8. 检查 Notebook 从干净 kernel 自上而下运行，不依赖手工执行顺序。

## 7. 交付边界

- 最终模型训练代码：只包含 Logistic MIL。
- 其他模型：只加载冻结预测/metrics，用于复现 PPT 第 6 页比较表，不提供训练代码。
- 早期 SVM：只记录路线与定性观察，不生成不可复现的成绩。
- OOD：只评估，绝不参与训练、早停、模型选择或阈值搜索。
- Notebook 完成后，需要把所有 PPT 数据表和数据图与 Notebook 产物逐项核对。

