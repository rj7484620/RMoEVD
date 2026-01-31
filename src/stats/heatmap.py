import numpy as np
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns


# read test_cwe.parquet
df = pd.read_parquet('../../data/test_cwe.parquet')

cwe_list = [
    "CWE-664",
    "CWE-unknown",# cwe-no-info in MoEVD
    "CWE-707",
    "CWE-399",
    "CWE-264",
    "CWE-691",
    "CWE-284",
    "CWE-682",
    "CWE-189",
    "CWE-other",
    "CWE-703",
    "CWE-254",
]
base_dir = '../../models/output_codebert-base_seed0'

md_lines = []

# ignore other cwe
f1_matrix_1 = []
for cwe_expert in cwe_list:
    npy_path = f"{base_dir}/{cwe_expert}/test_pred_proba.npy"
    data = np.load(npy_path)
    f1_list = []
    for cwe in cwe_list:
        tp, fn, tn, fp = 0, 0, 0, 0
        for i in range(len(df)):
            gt = df['level1'][i]
            if str(gt) == cwe:
                if data[i] > 0.5:
                    tp += 1
                else:
                    fn += 1
            elif str(gt) == 'None':
                if data[i] > 0.5:
                    fp += 1
                else:
                    tn += 1
            else:
                continue
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        f1_list.append(f1)
    f1_matrix_1.append(f1_list)
plt.figure(figsize=(10, 8))
sns.heatmap(f1_matrix_1, annot=True, fmt=".2f", xticklabels=cwe_list, yticklabels=cwe_list, cmap="YlGnBu")
plt.xlabel("CWE types")
plt.ylabel("Expert Models")
plt.title("F1 Score Heatmap")
plt.tight_layout()
plt.savefig("f1_heatmap_1.png")
plt.show()

