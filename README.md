# PML 2026 课程作业工作区

本工作区用于准备概率机器学习课程作业中蚊子音频检测的数据。

## 数据结构

- `data/raw/humbugdb/`: HumBugDB 0.0.1 数据集（来自 Zenodo 记录 4904800），包含四个 zip 压缩包和元数据 CSV 文件。
- `data/raw/ood_negative/data1/`: 70 个 OOD 负样本音频（y01.wav - y70.wav）。
- `data/raw/ood_negative/data2/`: 100 个 OOD 负样本音频，从 origin.wav 随机分割（y01.wav - y100.wav）。
- `data/raw/ood_positive/vasconcelos/`: 1697 个 Aedes Aegypti 蚊子音频样本，采样率 8kHz（Vasconcelos et al., 2020）。
- `data/raw/ood_positive/other/`: 其他 OOD 正样本（m1.wav - m10.wav）。

## 脚本

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

生成 HumBugDB 三种声音类型（蚊子、背景、音频）的时长分布直方图。

- `--input`: 元数据 CSV 路径（默认：`data/raw/humbugdb_0_0_1/neurips_2021_zenodo_0_0_1.csv`；当前布局使用 `--input data/raw/humbugdb/neurips_2021_zenodo_0_0_1.csv`）
- `--output`: 输出图像路径（默认：`outputs/humbugdb_label_duration_distribution.png`）
- `--bins`: 直方图分箱数量（默认：40）
- `--linear`: 使用线性 x 轴而非对数轴

## 数据来源

详见 `docs/data_sources.md` 获取详细的数据源 URL、许可证和数据使用说明。

## 环境搭建

```powershell
# 创建虚拟环境
uv venv

# 安装依赖
uv pip install -r requirements.txt
```

## 注意事项

- `data/raw/humbugdb/` 目录下的压缩包体积较大，未包含在仓库中，请自行从 Zenodo 下载并放入该目录。
- 所有脚本均支持直接从 zip 压缩包读取音频，无需手动解压。