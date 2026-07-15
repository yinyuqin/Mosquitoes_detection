# GPU MLP MIL 实验报告

## 1. 实验目的

本实验在 Logistic MIL 基线的基础上，只替换窗口分类器，以检验非线性模型是否能
提高 MFCC 窗口的判别能力。数据预处理、原始录音分组、训练/测试切分、五折交叉
验证、Top-3 聚合、损失函数和评估协议均保持不变。

## 2. 模型

MLP 输入仍是每个窗口 39 个 MFCC 通道的均值和标准差，共 78 维：

```text
78 → Linear(128) → LayerNorm → GELU → Dropout(0.2)
   → Linear(64)  → LayerNorm → GELU → Dropout(0.2)
   → Linear(1)   → window logit
```

模型共有 18,817 个参数。窗口概率使用 Top-3 平均聚合为片段概率；损失仍为类别
加权 bag BCE 加 `0.5 ×` 负 bag 全窗口背景 BCE。

## 3. 训练配置

| 配置 | 数值 |
|---|---:|
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU |
| hidden dimensions | 128, 64 |
| dropout | 0.2 |
| 优化器 | AdamW |
| 学习率 | 0.001 |
| weight decay | 0.0001 |
| batch size | 64 bags |
| 最大 epoch | 100 |
| 早停 patience | 12 |
| Top-k | 3 |
| 随机种子 | 2026 |
| OOF F1 最优阈值 | 0.20 |

五折最佳 epoch 为 10、17、13、17、14，对应最佳验证损失为 0.1566、0.1801、
0.2698、0.1475、0.1600。完整训练耗时 249.57 秒，约 4 分 10 秒。

## 4. 公平比较

MLP 与 Logistic 使用完全相同的 7,270 条五折训练池和 1,845 条独立测试集；两者
`test_indices.npy` 完全一致，训练与测试原始录音组交集为 0。

| 测试指标 | Logistic MIL | MLP MIL | MLP 变化 |
|---|---:|---:|---:|
| Accuracy | 90.79% | 94.09% | +3.31 pp |
| Balanced Accuracy | 91.13% | 92.93% | +1.80 pp |
| Precision | 96.83% | 96.49% | -0.35 pp |
| Recall / Sensitivity | 90.39% | 95.42% | +5.03 pp |
| Specificity | 91.87% | 90.45% | -1.42 pp |
| False Positive Rate | 8.13% | 9.55% | +1.42 pp |
| F1 | 93.50% | 95.95% | +2.45 pp |
| MCC | 0.7828 | 0.8506 | +0.0678 |
| ROC-AUC | 96.35% | 98.70% | +2.34 pp |
| Average Precision | 98.47% | 99.55% | +1.08 pp |
| Brier Score | 0.0816 | 0.0486 | -0.0330 |

MLP 测试集混淆矩阵：

|  | 预测背景 | 预测蚊子 |
|---|---:|---:|
| 真实背景 | TN = 445 | FP = 47 |
| 真实蚊子 | FN = 62 | TP = 1,291 |

相较 Logistic，MLP 将漏检从 130 条降至 62 条，但背景误报从 40 条增至 47 条。
总错误数从 170 条降至 109 条。

## 5. 结论与限制

MLP 在相同测试集上明显提高 Recall、F1、MCC、ROC-AUC 和 AP，说明 MFCC 窗口与
标签之间存在 Logistic 线性边界无法充分描述的非线性关系。Brier Score 下降也表明
当前 MLP 集成概率比 Logistic 更接近真实标签。

MLP 的 F1 最优阈值为 0.20，该工作点更偏向召回，导致 FPR 上升至 9.55%。不同模型
的概率阈值不能直接比较；实际部署应在训练 OOF 预测上选择相同目标 FPR，再比较可
达到的 Recall。较高 ROC-AUC 表明 MLP 可能在低 FPR 工作点继续保持优势，但需要
单独报告阈值曲线加以验证。

本实验仍是 HumBugDB 内部分组测试，不证明 OOD 或真实部署效果。下一步应在不参与
训练和调参的 OOD 正负数据上评估，并重点报告 FPR ≤ 1% 时的 Recall。

## 6. 复现

在仓库根目录执行：

```powershell
uv run python .\scripts\train_mlp_mil_gpu.py `
  --groups .\data\processed\humbugdb_mfcc\groups.npy
```

默认产物写入 `outputs/mlp_mil_gpu/`。训练产物和处理后数据不提交到 Git，实验参数
和结果由本报告记录。
