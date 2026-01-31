import numpy as np
import pandas as pd
from tabulate import tabulate

# 读取parquet文件
df = pd.read_parquet('/root/autodl-tmp/data/test_cwe.parquet')

cwe_list = [
    "CWE-664",
    "CWE-unknown",# cwe-no-info
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
base_dir = '/root/autodl-tmp/output_codebert-base_seed0'

md_lines = []
f1_matrix = []
values_list = []
for cwe in cwe_list:
    i_v, i_n = 0, 0
    j_v, j_n = 0, 0
    n_v, n_n = 0, 0

    tp, fn, tn, fp = 0, 0, 0, 0
    tp1, fn1, tn1, fp1 = 0, 0, 0, 0
    tp2, fn2, tn2, fp2 = 0, 0, 0, 0
    tp3, fn3, tn3, fp3 = 0, 0, 0, 0

    npy_path = f"{base_dir}/{cwe}/test_pred_proba.npy"
    data = np.load(npy_path)
    for i in range(len(df)):
        gt = df['level1'][i]
        if str(gt) == cwe:
            if data[i] > 0.5:
                i_v += 1
                tp += 1
                tp1 += 1
                tp2 += 1
                tp3 += 1
            else:
                i_n += 1
                fn += 1
                fn1 += 1
                fn2 += 1
                fn3 += 1
        elif str(gt) == 'None':
            if data[i] > 0.5:
                n_v += 1
                fp += 1
                fp2 += 1
                fp3 += 1
            else:
                n_n += 1
                tn += 1
                tn2 += 1
                tn3 += 1
        else:
            print(str(gt), " ", cwe)
            if data[i] > 0.5:
                j_v += 1
                fp1 += 1
                fp2 += 1
                tp3 += 1
            else:
                j_n += 1
                tn1 += 1
                tn2 += 1
                fn3 += 1
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
    f11 = 2 * tp1 / (2 * tp1 + fp1 + fn1) if (2 * tp1 + fp1 + fn1) > 0 else 0
    f12 = 2 * tp2 / (2 * tp2 + fp2 + fn2) if (2 * tp2 + fp2 + fn2) > 0 else 0
    f13 = 2 * tp3 / (2 * tp3 + fp3 + fn3) if (2 * tp3 + fp3 + fn3) > 0 else 0
    headers = [f"", f"", f"~~Vul~~ CWE-i", f"~~Non-Vul~~ Non-CWE-i"]
    values = [
        [f"**{cwe}**", "CWE-i", i_v, i_n],
        ["", "CWE-x", j_v, j_n],
        ["", "Non-Vul", n_v, n_n],
        ["","","",""],
    ]
    values_list.append(values[0])
    values_list.append(values[1])
    values_list.append(values[2])
    values_list.append(values[3])
    # md_lines.append(f"\n## {cwe}:")
    # md_lines.append(f"Calculate f1 on CWE-i and Non-Vul: **{f1:.4f}**")
    # md_lines.append(f"Calculate f1 on CWE-i and CWE-x: **{f11:.4f}**")
    # md_lines.append(f"Calculate f1 on All and treat CWE-x as negative sample: **{f12:.4f}**")
    # md_lines.append(f"Calculate f1 on ALL and treat CWE-x as positive sample: **{f13:.4f}**\n")
    # md_lines.append(tabulate(values, headers=headers, tablefmt="github"))
    f1_matrix.append([
        f"**{cwe}**",
        f"{f1:.4f}",
        f"{f11:.4f}",
        f"{f12:.4f}",
        f"{f13:.4f}"
    ])

headers = [f"", f"", f"~~Vul~~ CWE-i", f"~~Non-Vul~~ Non-CWE-i"]
md_lines.append(tabulate(values_list, headers=headers, tablefmt="github"))

headers = ['','Calculate f1 on CWE-i and Non-Vul', 
             'Calculate f1 on CWE-i and CWE-x',
             'Calculate f1 on All and treat CWE-x as negative sample',
             'Calculate f1 on All and treat CWE-x as positive sample']
md_lines.append("\n## F1 Matrix:")
md_lines.append(tabulate(f1_matrix, headers=headers, tablefmt="github"))

with open("statistic_result.md", "w", encoding="utf-8") as f:
    f.writelines(line + "\n" for line in md_lines)