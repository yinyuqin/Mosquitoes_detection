# HumBugDB 训练数据深度分析

输入：`E:/DataFiles/CodeLibrary/Agents/Codex/PML/data/raw/humbugdb_0_0_1/neurips_2021_zenodo_0_0_1.csv`  
记录行数：**9,295**；原始录音名：**3,237**。

## 1. 类别与时长

| sound_type | 片段数 | 占比 | 时长(h) | 中位数(s) | P95(s) |
|---|---|---|---|---|---|
| background | 1,900 | 20.4% | 14.630 | 5.021 | 59.648 |
| mosquito | 6,795 | 73.1% | 18.318 | 5.120 | 35.780 |
| audio | 600 | 6.5% | 0.513 | 2.445 | 6.935 |

说明：CSV 的每一行是一个标注片段；`name` 才是划分训练/验证/测试时应使用的最小分组键。

## 2. 数据完整性与泄漏风险

- 完全重复行（扣除首次出现）：**0**
- 重复 ID 数：**0**
- 同一录音名同时含多种标签：**128**
- 非法或非正时长行：**0**
- 高风险点：如果按 CSV 行随机切分，同一原始录音的片段会跨集合，导致明显的数据泄漏。必须按 `name` 分组切分；更严格时再按录制日期、地点和设备组成 session 分组。

### 最大的录音分组

| 录音名 | 片段数 |
|---|---|
| R4_cleaned recording_TEL_25-10-17.wav | 211 |
| R4_cleaned recording_TEL_24-10-17.wav | 210 |
| R4_cleaned recording_TEL_23-10-17.wav | 179 |
| R4_cleaned recording_TEL_20-10-17.wav | 141 |
| #423-487.wav | 124 |
| R4_cleaned recording_TEL_19-10-17.wav | 117 |
| R4_cleaned recording_18-10-17.wav | 98 |
| R4_cleaned recording_17-10-17.wav | 83 |
| CDC_Ae-albopictus_labelled_800.wav | 79 |
| R4_cleaned recording_16-10-17.wav | 76 |
| #540-576.wav | 72 |
| #658-696.wav | 66 |
| CDC_An-quadrimaculatus_labelled_800.wav | 64 |
| #1335-1369.wav | 64 |
| #26-60.wav | 56 |

## 3. 元数据缺失率

| 字段 | 缺失数 | 缺失率 |
|---|---|---|
| age | 8989 | 96.7% |
| fed | 5641 | 60.7% |
| gender | 5590 | 60.1% |
| plurality | 4455 | 47.9% |
| species | 3287 | 35.4% |
| method | 3254 | 35.0% |
| province | 1381 | 14.9% |
| district | 1376 | 14.8% |
| device_type | 1183 | 12.7% |
| id | 0 | 0.0% |
| length | 0 | 0.0% |
| name | 0 | 0.0% |
| sample_rate | 0 | 0.0% |
| record_datetime | 0 | 0.0% |
| sound_type | 0 | 0.0% |
| mic_type | 0 | 0.0% |
| country | 0 | 0.0% |
| place | 0 | 0.0% |
| location_type | 0 | 0.0% |

## 4. 主要分布

### sample_rate

| 值 | 片段数 | 占比 |
|---|---|---|
| 44100 | 6102 | 65.6% |
| 8000 | 3193 | 34.4% |

### species

| 值 | 片段数 | 占比 |
|---|---|---|
| <missing> | 3287 | 35.4% |
| an arabiensis | 1985 | 21.4% |
| an gambiae ss | 737 | 7.9% |
| culex quinquefasciatus | 678 | 7.3% |
| culex pipiens complex | 545 | 5.9% |
| an funestus ss | 381 | 4.1% |
| an squamosus | 141 | 1.5% |
| ma uniformis | 131 | 1.4% |
| an dirus | 129 | 1.4% |
| an harrisoni | 124 | 1.3% |
| ae aegypti | 123 | 1.3% |
| an maculatus | 117 | 1.3% |
| an funestus sl | 104 | 1.1% |
| an coustani | 92 | 1.0% |
| ae albopictus | 79 | 0.8% |

### gender

| 值 | 片段数 | 占比 |
|---|---|---|
| <missing> | 5590 | 60.1% |
| Female | 3655 | 39.3% |
| Male | 50 | 0.5% |

### plurality

| 值 | 片段数 | 占比 |
|---|---|---|
| Single | 4812 | 51.8% |
| <missing> | 4455 | 47.9% |
| Plural | 28 | 0.3% |

### method

| 值 | 片段数 | 占比 |
|---|---|---|
| <missing> | 3254 | 35.0% |
| HBN | 2123 | 22.8% |
| ABN | 1745 | 18.8% |
| LT | 1609 | 17.3% |
| LC | 378 | 4.1% |
| HLC | 186 | 2.0% |

### mic_type

| 值 | 片段数 | 占比 |
|---|---|---|
| telinga | 4919 | 52.9% |
| phone | 3193 | 34.4% |
| Telinga | 1183 | 12.7% |

### device_type

| 值 | 片段数 | 占比 |
|---|---|---|
| tascam | 2673 | 28.8% |
| olympus | 2246 | 24.2% |
| Alcatel 4015X | 1335 | 14.4% |
| <missing> | 1183 | 12.7% |
| itel A16 | 1163 | 12.5% |
| Alcatel 4009X | 538 | 5.8% |
| Alcatel | 157 | 1.7% |

### country

| 值 | 片段数 | 占比 |
|---|---|---|
| Tanzania | 3836 | 41.3% |
| Thailand | 2205 | 23.7% |
| UK | 1381 | 14.9% |
| Kenya | 1335 | 14.4% |
| USA | 538 | 5.8% |

### district

| 值 | 片段数 | 占比 |
|---|---|---|
| Kilombero District | 3836 | 41.3% |
| Sai Yok District | 2205 | 23.7% |
| <missing> | 1376 | 14.8% |
| Oxfordshire | 1340 | 14.4% |
| Georgia | 538 | 5.8% |

### province

| 值 | 片段数 | 占比 |
|---|---|---|
| Morogoro | 3836 | 41.3% |
| Kanchanaburi Province | 2205 | 23.7% |
| <missing> | 1381 | 14.9% |
| Nairobi | 1335 | 14.4% |
| Atlanta | 538 | 5.8% |

### place

| 值 | 片段数 | 占比 |
|---|---|---|
| Ifakara | 3836 | 41.3% |
| field site near Pu Teuy Village | 2205 | 23.7% |
| Oxford Zoology | 1340 | 14.4% |
| USAMRU-K | 1335 | 14.4% |
| CDC insect cultures, Atlanta | 538 | 5.8% |
| LSTMH Culture cages, London | 41 | 0.4% |

### location_type

| 值 | 片段数 | 占比 |
|---|---|---|
| cup | 6061 | 65.2% |
| culture | 2071 | 22.3% |
| field | 1163 | 12.5% |

## 5. 外部/Freesound 重合检查

本次未提供外部候选元数据，因此不能直接判断 Freesound 是否与 HumBugDB 重合。可用 `--external-metadata freesound.csv` 做第一轮文件名筛查。

## 6. 建模建议

1. 按 `name` 做 GroupKFold/GroupShuffleSplit，禁止逐行随机切分。
2. 分别报告按片段与按录音聚合的指标，避免大录音贡献过多权重。
3. 对 sample rate、设备、国家和地点做分层审计；OOD 测试应尽量选择训练中未出现的设备/环境。
4. 类别权重同时参考片段数和总时长；长背景片段不应直接压倒短蚊子片段。
5. Freesound 数据保持为只读 OOD 测试集，不参与阈值选择，并保留作者、URL、许可证和真实性审核字段。
