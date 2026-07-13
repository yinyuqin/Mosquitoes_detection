# 蚊子音频检测 - PML 2026 大作业

本仓库包含概率机器学习（PML）课程 2026 年大作业的完整代码和文档，目标是使用音频数据构建蚊子声音检测的机器学习分类模型。

## 项目目标

构建一个具有实际使用价值的蚊子检测器，能够：

- 对真实蚊子声音保持较高的敏感度/召回率；
- 对日常背景声音保持较低的误报率；
- 在风声、交通、人声、音乐、其他昆虫声等现实环境中进行测试和分析；
- **核心挑战**：证明模型能泛化到分布外数据（OOD），即来源、设备或录制环境与训练集不同的音频。

## 数据结构

### 主数据集：HumBugDB

- `data/raw/humbugdb/`: HumBugDB 0.0.1 数据集（来自 Zenodo 记录 4904800），包含四个 zip 压缩包和元数据 CSV 文件。

标签映射：
- `mosquito`：正类，表示存在蚊子声音；
- `audio`：负类/背景类；
- `background`：负类/背景类。

### OOD 数据集

- `data/raw/ood_negative/data1/`: 70 个 OOD 负样本音频（日常背景声音）；
- `data/raw/ood_negative/data2/`: 100 个 OOD 负样本音频，从 origin.wav 随机分割；
- `data/raw/ood_positive/vasconcelos/`: 1697 个 Aedes Aegypti 蚊子音频样本，采样率 8kHz（Vasconcelos et al., 2020）；
- `data/raw/ood_positive/other/`: 其他 OOD 正样本。

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
# 创建虚拟环境
uv venv

# 安装依赖
uv pip install -r requirements.txt
```

## 文档

- `docs/requirements.md`: 大作业实验要求概括；
- `docs/data_sources.md`: 数据来源、许可证和数据使用说明；
- `docs/analysis/dataset_analysis.md`: 数据集分析报告；
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