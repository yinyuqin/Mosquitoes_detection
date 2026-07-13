# PML 课程作业数据来源

## 主要分布内数据集

使用 Zenodo 记录 4904800 中的 HumBugDB 0.0.1 作为主要训练和验证数据源：

- `neurips_2021_zenodo_0_0_1.csv`
- `humbugdb_neurips_2021_1.zip`
- `humbugdb_neurips_2021_2.zip`
- `humbugdb_neurips_2021_3.zip`
- `humbugdb_neurips_2021_4.zip`

课程作业 PDF 中将蚊子存在标记为 `sound type` 字段；在下载的 CSV 中，该列名为 `sound_type`：

- `mosquito` -> 正类（蚊子）
- `audio` 或 `background` -> 负类/背景类

本地存储位置：`data/raw/humbugdb/`

## OOD 负样本/背景数据

当前 OOD 负样本目录：

- `data/raw/ood_negative/data1/` - 包含 70 个音频样本（y01.wav - y70.wav）
- `data/raw/ood_negative/data2/` - 包含 100 个音频样本（y01.wav - y100.wav）

## OOD 正样本蚊子数据

使用 Vasconcelos et al. (2020) 作为公共外部 OOD 正样本子集：

- 数据集：*Mosquitoes Bioacoustic Features - A Public Dataset*
- 来源：<https://doi.org/10.6084/m9.figshare.11902125>
- 本地归档：`data/raw/ood_positive_vasconcelos_2020/Mosquitos Audio Samples.rar`
- 许可证：CC BY 4.0
- 用途：仅用于外部蚊子正样本 OOD 评估；不提供 OOD 负样本
- 质量注意：评估前排除零时长 WAV 文件

当前已整理的 Vasconcelos 数据集：

- `data/raw/ood_positive/vasconcelos/` - 包含 1697 个 8kHz 采样率的音频文件

其他 OOD 正样本数据：

- `data/raw/ood_positive/other/` - 其他正样本数据

