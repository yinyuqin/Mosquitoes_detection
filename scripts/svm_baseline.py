import numpy as np
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, average_precision_score
)
from sklearn.preprocessing import StandardScaler
import time
import joblib

# --------------------------
# 1. 加载预处理好的数据
# --------------------------
print("正在加载数据...")
features_path = 'data/processed/humbugdb_mfcc/mfcc_features.npy'
labels_path = 'data/processed/humbugdb_mfcc/labels.npy'
meta_path = 'data/processed/humbugdb_mfcc/metadata.json'

mfcc_data = np.load(features_path, allow_pickle=True)
labels = np.load(labels_path)

with open(meta_path, 'r') as f:
    metadata = json.load(f)

print(f"数据加载完成。样本总数: {len(mfcc_data)}, 标签分布: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}")

# --------------------------
# 2. 特征工程：方案 A - 统计聚合
# --------------------------
print("正在进行特征降维 (Mean + Std)...")
fixed_dim_features = []

for i in range(len(mfcc_data)):
    sample = mfcc_data[i]
    if isinstance(sample, np.ndarray):
        mean_feat = np.mean(sample, axis=1)
        std_feat = np.std(sample, axis=1)
        combined_feat = np.concatenate([mean_feat, std_feat])
        fixed_dim_features.append(combined_feat)
    else:
        fixed_dim_features.append(np.zeros(78))

X = np.array(fixed_dim_features)
y = labels.astype(int)

# --------------------------
# 3. 数据集划分
# --------------------------
print("正在划分数据集 (Training 80%, Validation 20%)...")
# test_size=0.2 表示验证集占20%，训练集占80%，即 4:1 比例
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=2026, stratify=y
)

# --------------------------
# 4. 数据标准化
# --------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# --------------------------
# 5. 训练 SVM 模型
# --------------------------
print("开始训练 SVM Baseline...")
start_time = time.time()

svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced', random_state=42)
svm_model.fit(X_train_scaled, y_train)

end_time = time.time()
print(f"SVM 训练完成。耗时: {end_time - start_time:.2f} 秒")

# --------------------------
# 6. 评估 (在 Validation 集上输出所有指定指标)
# --------------------------
y_pred = svm_model.predict(X_val_scaled)
y_prob = svm_model.predict_proba(X_val_scaled)[:, 1]

# 计算各项指标
acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
cm = confusion_matrix(y_val, y_pred)
roc_auc = roc_auc_score(y_val, y_prob)
pr_auc = average_precision_score(y_val, y_prob)

# 从混淆矩阵中提取 TN, FP, FN, TP 并计算 FPR
tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn)

# 打印格式化结果
print("\n" + "="*40)
print("--- SVM Baseline 验证集 (Validation) 测试结果 ---")
print("="*40)
print(f"1. Accuracy (准确率):               {acc:.4f}")
print(f"2. Precision (精确率):              {prec:.4f}")
print(f"3. Recall/Sensitivity (召回率/灵敏度): {rec:.4f}")
print(f"4. F1 Score:                        {f1:.4f}")
print(f"5. False Positive Rate (假阳性率):  {fpr:.4f}")
print(f"6. ROC-AUC:                         {roc_auc:.4f}")
print(f"7. PR-AUC (Average Precision):      {pr_auc:.4f}")

print("\n8. Confusion Matrix (混淆矩阵):")
print(cm)
print(f"-> [[TN={tn}, FP={fp}]\n   [FN={fn}, TP={tp}]]")
print("="*40 + "\n")

# --------------------------
# 7. 保存模型 (自动创建目录)
# --------------------------
os.makedirs('models', exist_ok=True)
joblib.dump(svm_model, 'models/svm_baseline.pkl')
joblib.dump(scaler, 'models/scaler_baseline.pkl')
print("模型与 Scaler 已保存至 models/ 目录")