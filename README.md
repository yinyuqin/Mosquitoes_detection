# 蚊子音频检测 - PML 2026 大作业

本仓库包含概率机器学习（PML）课程 2026 年大作业的完整代码和文档，目标是使用音频数据构建蚊子声音检测的机器学习分类模型。

## 项目目标

构建一个具有实际使用价值的蚊子检测器，能够：

- 对真实蚊子声音保持较高的敏感度/召回率；
- 对日常背景声音保持较低的误报率；
- 在风声、交通、人声、音乐、其他昆虫声等现实环境中进行测试和分析；
- **核心评估目标**：验证模型能否泛化到分布外数据（OOD），即来源、设备或录制环境与训练集不同的音频。

## 数据结构

### 主数据集：HumBugDB

- `data/raw/humbugdb/`: HumBugDB 0.0.1 数据集（来自 Zenodo 记录 4904800），包含四个 zip 压缩包和元数据 CSV 文件。

标签映射：
- `mosquito`：正类，表示存在蚊子声音；
- `audio`：负类/背景类；
- `background`：负类/背景类。

### OOD 数据集

- `data/raw/ood_negative/data1.zip`: 70 个 OOD 负样本音频（日常背景声音）；
- `data/raw/ood_negative/data2.zip`: 100 个 OOD 负样本音频，从同一个 `origin.wav` 随机分割，评估时应按源录音视为同一组；
- `data/raw/ood_positive/vasconcelos.zip`: 1697 个 Aedes Aegypti 蚊子音频样本，采样率 8kHz（Vasconcelos et al., 2020）；
- `data/raw/ood_positive/other.zip`: 其他 OOD 正样本。

## 脚本说明

### 处理 HumBugDB 并提取 MFCC 特征

```powershell
uv run python .\scripts\process_humbugdb_from_zip.py
```

从 HumBugDB 的 zip 文件中处理音频并提取 MFCC 特征。支持以下参数配置：

- `--zip-dir`: humbugdb zip 文件所在目录（默认：`data/raw/humbugdb/`）
- `--csv`: 元数据 CSV 文件路径（默认：`data/raw/humbugdb/neurips_2021_zenodo_0_0_1.csv`）
- `--output`: 处理后的 MFCC 特征输出目录（默认：`data/processed/humbugdb_mfcc/`）
- `--n-mfcc`: MFCC 系数数量（默认：13）
- `--sample-rate`: 目标采样率（默认：8000）

预处理按 CSV 中的 `id` 读取对应的 `id.wav`，统一重采样到 8 kHz，并提取
MFCC、Delta 和 Delta² 共 39 个通道。默认过滤短于 0.1 秒或长于 60 秒的片段。
当前完整处理结果为 9,115 个片段，其中蚊子 6,683 个、背景 2,432 个；另有
180 个片段因时长限制被过滤。

除特征和标签外，脚本还会保存：

- `groups.npy`：CSV 的 `name` 字段，用于原始录音级分组切分；
- `sample_ids.npy`：CSV 的唯一片段 ID，用于结果追踪。

### 训练 GPU Logistic MIL 基线

```powershell
uv run python .\scripts\train_logistic_mil_gpu.py `
  --groups .\data\processed\humbugdb_mfcc\groups.npy
```

训练脚本使用 CUDA PyTorch。每个 MFCC 片段被视为一个由多个窗口组成的 bag：

- 窗口为 64 帧，步长为 32 帧，即 50% 重叠；
- 短片段训练时随机进行左右补零，验证和测试时居中补零；
- 每个窗口以 39 个通道的均值和标准差组成 78 维特征；
- 线性 Logistic 模型输出窗口概率，Top-3 平均得到片段概率；
- 正负样本使用二元交叉熵，负 bag 另有全窗口背景损失；
- 先按原始录音保留 20% 独立测试集，再在训练集上做五折分组交叉验证；
- 分类阈值只根据训练集 OOF 预测确定，测试集不参与调参。

RTX 4060 上的当前实验使用阈值 0.32，在原始录音隔离的 HumBugDB 测试集上得到：

| 指标 | 测试结果 |
|---|---:|
| Accuracy | 90.79% |
| Balanced Accuracy | 91.13% |
| Precision | 96.83% |
| Recall/Sensitivity | 90.39% |
| Specificity | 91.87% |
| False Positive Rate | 8.13% |
| F1 | 93.50% |
| ROC-AUC | 96.35% |
| Average Precision (AP) | 98.47% |

这些结果是 HumBugDB 内部分组测试结果，不代表 OOD 或真实部署表现。完整实验协议、
混淆矩阵和局限性见
[`docs/experiments/logistic_mil_gpu.md`](docs/experiments/logistic_mil_gpu.md)。

### 绘制标签时长分布

```powershell
uv run python .\scripts\plot_label_duration_distribution.py
```

生成 HumBugDB 三种声音类型的时长分布直方图。

### 压缩 WAV 文件

```powershell
uv run python .\scripts\compress_wav_files.py
```

将 WAV 文件压缩为 zip 归档以节省空间。

## 环境搭建

```powershell
# 以下命令均在仓库根目录执行

# 创建虚拟环境
uv venv

# 安装依赖
uv pip install -r requirements.txt
```

当前依赖固定到 PyTorch 2.12.1 CUDA 13.0 构建。安装 CUDA wheel 需要约 1.8 GiB
下载空间；运行训练前应确认 `torch.cuda.is_available()` 为 `True`。

训练脚本默认要求 CUDA，不会静默回退。需要进行 CPU 功能测试时，可显式添加
`--device cpu`；该模式不代表本报告使用的训练环境。

## 文档

- `docs/requirements.md`: 大作业实验要求概括；
- `docs/data_sources.md`: 数据来源、许可证和数据使用说明；
- `docs/analysis/dataset_analysis.md`: 数据集分析报告；
- `docs/experiments/logistic_mil_gpu.md`: GPU Logistic MIL 基线实验报告；
- `docs/PML_2026_coursework.pdf`: 课程作业官方要求文档。

## 评分标准

本大作业评分分为四个部分：

| 部分 | 分数 |
|------|------|
| 项目介绍 | 20/100 |
| 问题分析 | 20/100 |
| 数据收集工作 | 20/100 |
| 方法实现与评估 | 40/100 |
| 英文展示 Bonus | 5/100 |

## 重要说明

- `data/raw/humbugdb/` 目录下的压缩包体积较大（约 4GB），未包含在仓库中，请自行从 [Zenodo](https://zenodo.org/records/4904800) 下载并放入该目录；
- 所有脚本均支持直接从 zip 压缩包读取音频，无需手动解压；
- OOD 测试集不能用于训练或调参，仅用于评估模型的泛化能力；
- 推荐评价指标：accuracy、precision、recall/sensitivity、F1 score、confusion matrix、false positive rate、ROC-AUC 或 PR-AUC。

## 参考文献

Ivan Kiskin et al. *HumBugDB: a large-scale acoustic mosquito dataset*. NeurIPS 2021 Track on Datasets and Benchmarks, 2021.
