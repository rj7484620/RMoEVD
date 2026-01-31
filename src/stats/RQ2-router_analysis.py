import pandas as pd
import numpy as np

# ================================
# 1. Load model prediction probabilities
# ================================
pred_path = "../../models/output_codebert-base_seed42/focal_ep20_bs512_eval_f1_macro_gamma1/testset_pred_proba.pkl"
pred_df = pd.read_pickle(pred_path)   # Each row = one sample, each column = probability of a CWE class

print("Data type:", type(pred_df))
print("Data shape:", pred_df.shape)
print("Preview:")
print(pred_df.head())

# Convert to numpy array for faster processing
pred_array = pred_df.values   # shape: (num_samples, num_classes)

# ================================
# 2. Define CWE class order
# IMPORTANT: This order must match the column order of pred_df
# ================================
cwe_list = [
    "CWE-189", "CWE-254", "CWE-264", "CWE-284",
    "CWE-399", "CWE-664", "CWE-682", "CWE-691",
    "CWE-703", "CWE-707", "CWE-other", "CWE-unknown"
]

# Counters
# cnt_top1: counts when a class is Top-1 prediction for benign samples
# cnt_top2: counts when a class appears in Top-2 predictions for benign samples
cnt_top1 = [0] * len(cwe_list)
cnt_top2 = [0] * len(cwe_list)

# ================================
# 3. Load ground-truth test labels
# ================================
label_path = "../../data/test_cwe.parquet"
label_df = pd.read_parquet(label_path)

# ================================
# 4. Iterate through samples
# ================================
for i in range(len(pred_array)):
    probs = pred_array[i]

    # Get indices of Top-2 predicted classes (sorted descending)
    top2_indices = np.argsort(probs)[-2:][::-1]
    top1_idx, top2_idx = top2_indices

    print(f"Sample {i}: Top1={cwe_list[top1_idx]}, Top2={cwe_list[top2_idx]}")

    # Only analyze samples whose ground-truth label is 'None' (benign / non-vulnerable)
    if str(label_df['level1'][i]) == 'None':
        # Count Top-1 misclassification
        cnt_top1[top1_idx] += 1

        # Count both Top-1 and Top-2 for Top-2 statistics
        cnt_top2[top1_idx] += 1
        cnt_top2[top2_idx] += 1

# ================================
# 5. Final statistics
# ================================
print("\nCWE Classes:")
print(cwe_list)

print("\nTop-1 prediction counts for benign samples:")
print(cnt_top1)

print("\nTop-2 prediction counts for benign samples:")
print(cnt_top2)
