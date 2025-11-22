import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

# -------------------------------------------------------------------------
# --- USER INPUT -----------------------------------------------------------
# -------------------------------------------------------------------------
csv_gt = "GT.csv"      # Ground Truth CSV
csv_pred = "Prediction.csv"     # Predictions CSV

id_col_gt = "id"                 # ID column name in ground truth file
label_col_gt = "Classification_GT"        # Ground truth label column (0,1,2,3)

id_col_pred = "id"
label_col_pred = "Classification_result "   # Predicted label column (0,1,2,3)
# -------------------------------------------------------------------------

# Read the CSVs
df_gt = pd.read_csv(csv_gt)
df_pred = pd.read_csv(csv_pred)

print("Prediction file columns:", df_pred.columns.tolist())

# Select only required columns
df_gt = df_gt[[id_col_gt, label_col_gt]]
df_pred = df_pred[[id_col_pred, label_col_pred]]

# Merge using the ID column
df = pd.merge(df_gt, df_pred,
              left_on=id_col_gt,
              right_on=id_col_pred,
              how="inner")

y_true = df[label_col_gt].astype(int)
y_pred = df[label_col_pred].astype(int)

# ---------------------------------------------------------
# 1. TOP-1 ACCURACY
# ---------------------------------------------------------
top1_accuracy = accuracy_score(y_true, y_pred)

# ---------------------------------------------------------
# 2. F1 SCORE (macro)
# ---------------------------------------------------------
f1_macro = f1_score(y_true, y_pred, average="macro")

# ---------------------------------------------------------
# 3. MULTICLASS AUC (macro, using One-vs-Rest)
# ---------------------------------------------------------
# Convert to one-hot for AUC
y_true_oh = pd.get_dummies(y_true)

# If prediction probabilities are NOT available,
# convert class labels into one-hot (not ideal but works)
y_pred_oh = pd.get_dummies(y_pred)

auc_macro = roc_auc_score(y_true_oh, y_pred_oh, average="macro")

# ---------------------------------------------------------
# 4. Sensitivity, Specificity, PPV per class
# ---------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)  # 4x4 matrix
num_classes = cm.shape[0]

sensitivity = {}
specificity = {}
ppv = {}

for cls in range(num_classes):
    TP = cm[cls, cls]
    FN = cm[cls, :].sum() - TP
    FP = cm[:, cls].sum() - TP
    TN = cm.sum() - (TP + FP + FN)

    sensitivity[cls] = TP / (TP + FN) if (TP + FN) > 0 else 0
    specificity[cls] = TN / (TN + FP) if (TN + FP) > 0 else 0
    ppv[cls] = TP / (TP + FP) if (TP + FP) > 0 else 0

# ---------------------------------------------------------
# PRINT RESULTS
# ---------------------------------------------------------
print("\n===== MULTICLASS METRICS =====")
print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
print(f"F1 Score (Macro): {f1_macro:.4f}")
print(f"AUC (Macro): {auc_macro:.4f}")

print("\n--- Per-Class Metrics (0,1,2,3) ---")
for cls in range(4):
    print(f"\nClass {cls}:")
    print(f"  Sensitivity (Recall): {sensitivity[cls]:.4f}")
    print(f"  Specificity:          {specificity[cls]:.4f}")
    print(f"  PPV (Precision):      {ppv[cls]:.4f}")
