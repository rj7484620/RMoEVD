import pandas as pd

file_path = "/root/autodl-tmp/output_codebert-base_seed42/focal_ep20_bs512_eval_f1_macro_gamma1/testset_pred_proba.pkl"

df = pd.read_pickle(file_path)

print("数据类型:", type(df))
print("数据维度:", df.shape if hasattr(df, "shape") else "无 shape 属性")
print("前几行数据:")
print(df.head())

import pandas as pd

file_path = "/root/autodl-tmp/output_codebert-base_seed42/focal_ep20_bs512_eval_f1_macro_gamma1/testset_pred_proba.pkl"

ddf = pd.read_pickle(file_path)

cwe_list = ["CWE-189", "CWE-254", "CWE-264", "CWE-284", "CWE-399", "CWE-664", "CWE-682", "CWE-691", "CWE-703", "CWE-707", "CWE-other", "CWE-unknown"]
cnt_list_1 = [0] * len(cwe_list)
cnt_list_2 = [0] * len(cwe_list)
# 转为二维list，丢弃表头
data_2d_list = ddf.values.tolist()
df = pd.read_parquet('/root/autodl-tmp/data/test_cwe.parquet')

for i in range(len(data_2d_list)):
    # find max 的索引 in data_2d_list[i]
   max_index = data_2d_list[i].index(max(data_2d_list[i]))
   print(f"第 {i} 行的最大值索引: {max_index}, 对应的 CWE: {cwe_list[max_index]}")
   # find 第二大的索引 in data_2d_list[i]
   second_max_index = data_2d_list[i].index(sorted(data_2d_list[i], reverse=True)[1])
   print(f"第 {i} 行的第二大值索引: {second_max_index}, 对应的 CWE: {cwe_list[second_max_index]}")
   # find
   if str(df['level1'][i]) == 'None':
      cnt_list_1[max_index] += 1
      cnt_list_2[max_index] += 1
      cnt_list_2[second_max_index] += 1

print(cwe_list)
print(cnt_list_1) 
print(cnt_list_2) 
