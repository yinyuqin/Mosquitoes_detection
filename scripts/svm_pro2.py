import numpy as np
import json
import os
import warnings
import time
import joblib

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_auc_score, average_precision_score
)

warnings.filterwarnings('ignore', category=FutureWarning)

# ================= 配置参数 =================
WINDOW_SIZE = 64       # 窗口帧数
K = 3                  # Top-K 窗口数量 (训练和推理时保持一致)
FEATURE_DIM = 78       # 单个窗口的特征维度 (Mean 39 + Std 39)

np.random.seed(42)

# ================= 辅助函数：切分窗口并提取特征 =================
def extract_window_features(mfcc_sample):
    n_mfcc, n_frames = mfcc_sample.shape
    num_windows = n_frames // WINDOW_SIZE
    windows_feat = []
    for w in range(num_windows):
        window_data = mfcc_sample[:, w * WINDOW_SIZE : (w + 1) * WINDOW_SIZE]
        mean_feat = np.mean(window_data, axis=1)
        std_feat = np.std(window_data, axis=1)
        windows_feat.append(np.concatenate([mean_feat, std_feat]))
    return np.array(windows_feat) if windows_feat else np.zeros((0, FEATURE_DIM))

# ================= 1. 加载数据与全局 Baseline 模型 =================
print("正在加载数据与全局 Baseline 模型...")
mfcc_data = np.load('data/processed/humbugdb_mfcc/mfcc_features.npy', allow_pickle=True)
labels = np.load('data/processed/humbugdb_mfcc/labels.npy')

svm_global = joblib.load('models/svm_baseline.pkl')
scaler_global = joblib.load('models/scaler_baseline.pkl')

print(f"数据加载完成。样本总数: {len(mfcc_data)}, 标签分布: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}")

# ================= 2. 核心：先在【音频级别】划分数据集 =================
print("正在划分数据集 (Training 80%, Validation 20%)...")
audio_indices = np.arange(len(mfcc_data))
train_idx, val_idx = train_test_split(
    audio_indices, test_size=0.2, random_state=42, stratify=labels
)
print(f"训练集音频数: {len(train_idx)}, 验证集音频数: {len(val_idx)}")

# ================= 3. 构建【窗口级别】的训练集 =================
print(f"正在为训练集构建 Top-{K} 窗口级训练数据 (正样本Top-K, 负样本随机)...")
X_train_windows = []
y_train_windows = []

for i in train_idx:
    windows_feat = extract_window_features(mfcc_data[i])
    if len(windows_feat) == 0: continue
        
    label = labels[i]
    num_w = len(windows_feat)
    actual_k = min(K, num_w)
    
    if label == 1:
        # 正样本：选最像蚊子的 Top-K
        windows_scaled = scaler_global.transform(windows_feat)
        probs = svm_global.predict_proba(windows_scaled)[:, 1]
        sorted_indices = np.argsort(probs)[::-1]
        selected_indices = sorted_indices[:actual_k]
    else:
        # 负样本：随机选 K 个
        selected_indices = np.random.choice(num_w, size=actual_k, replace=False)
    
    X_train_windows.extend(windows_feat[selected_indices])
    y_train_windows.extend([label] * actual_k)

X_train_windows = np.array(X_train_windows)
y_train_windows = np.array(y_train_windows)

# ================= 4. 标准化与训练【窗口级】 SVM =================
print("正在标准化并训练 窗口级 SVM...")
start_time = time.time()

scaler_win = StandardScaler()
X_train_scaled = scaler_win.fit_transform(X_train_windows)

svm_win = SVC(
    kernel='rbf', C=1.0, gamma='scale', probability=True, 
    class_weight='balanced', random_state=42
)
svm_win.fit(X_train_scaled, y_train_windows)

print(f"窗口级 SVM 训练完成。耗时: {time.time() - start_time:.2f} 秒")

# ================= 5. 验证集评估 (核心改进：Top-K Mean Pooling) =================
print("正在验证集上进行推理...")
all_window_probs_per_audio = []
y_true_audio = []

for i in val_idx:
    windows_feat = extract_window_features(mfcc_data[i])
    if len(windows_feat) == 0: continue
        
    windows_scaled = scaler_win.transform(windows_feat)
    window_probs = svm_win.predict_proba(windows_scaled)[:, 1]
    
    all_window_probs_per_audio.append(window_probs)
    y_true_audio.append(labels[i])

y_true_audio = np.array(y_true_audio)

# --- 🌟 核心改进：推理时使用 Top-K Mean Pooling ---
# 对每个音频，找出概率最高的 K 个窗口，计算它们的平均概率作为最终概率
y_prob_audio = []
for probs in all_window_probs_per_audio:
    actual_k = min(K, len(probs))
    # 降序排列，取前 K 个
    top_k_probs = np.sort(probs)[::-1][:actual_k]
    # 计算均值 (替代之前的 np.max)
    mean_prob = np.mean(top_k_probs)
    y_prob_audio.append(mean_prob)

y_prob_audio = np.array(y_prob_audio)

# --- 在验证集上搜索最佳阈值 ---
print("正在搜索最佳分类阈值 (目标：最大化 F1 Score)...")
thresholds = np.arange(0.01, 1.00, 0.01)
best_f1 = -1
best_threshold = 0.5

for thresh in thresholds:
    y_pred_temp = (y_prob_audio >= thresh).astype(int)
    f1_temp = f1_score(y_true_audio, y_pred_temp, zero_division=0)
    if f1_temp > best_f1:
        best_f1 = f1_temp
        best_threshold = thresh

print(f"👉 找到的最佳阈值为: {best_threshold:.2f} (对应最高 F1: {best_f1:.4f})\n")

# 使用最佳阈值计算最终的全套指标
y_pred_audio = (y_prob_audio >= best_threshold).astype(int)

acc = accuracy_score(y_true_audio, y_pred_audio)
prec = precision_score(y_true_audio, y_pred_audio, zero_division=0)
rec = recall_score(y_true_audio, y_pred_audio)
f1 = f1_score(y_true_audio, y_pred_audio)
cm = confusion_matrix(y_true_audio, y_pred_audio)
roc_auc = roc_auc_score(y_true_audio, y_prob_audio)
pr_auc = average_precision_score(y_true_audio, y_prob_audio)

tn, fp, fn, tp = cm.ravel()
fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

# 打印格式化结果
print("="*60)
print(f"--- 窗口级 Top-{K} SVM (推理聚合: Top-K Mean) 最终结果 ---")
print(f"--- 阈值={best_threshold:.2f} ---")
print("="*60)
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
print("="*60 + "\n")

# ================= 6. 保存最终模型与配置 =================
os.makedirs('models', exist_ok=True)
joblib.dump(svm_win, f'models/svm_window_topk{K}_mean_pool.pkl')
joblib.dump(scaler_win, f'models/scaler_window_topk{K}_mean_pool.pkl')

# 保存配置，记录聚合方式和最佳阈值
config = {'best_threshold': float(best_threshold), 'aggregation': 'top_k_mean', 'K': K}
joblib.dump(config, f'models/config_window_topk{K}_mean_pool.pkl')

print(f"模型、Scaler 及最佳阈值配置已保存至 models/ 目录")