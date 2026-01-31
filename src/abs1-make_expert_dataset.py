import pandas as pd

file_path = "/root/autodl-tmp/output_codebert-base_seed42/focal_ep20_bs512_eval_f1_macro_gamma1/testset_pred_proba.pkl"

df = pd.read_pickle(file_path)

print("数据类型:", type(df))
print("数据维度:", df.shape if hasattr(df, "shape") else "无 shape 属性")
print("前几行数据:")
print(df.head())


cwe_list = ["CWE-189", "CWE-254", "CWE-264", "CWE-284", "CWE-399", "CWE-664", "CWE-682", "CWE-691", "CWE-703", "CWE-707", "CWE-other", "CWE-unknown"]
# # 转为二维list，丢弃表头
# data_2d_list = df.values.tolist()

test_par = pd.read_parquet("/root/autodl-tmp/data/test_cwe.parquet")
print("数据类型:", type(test_par))
print("数据维度:", test_par.shape if hasattr(test_par, "shape") else "无 shape 属性")
print("前几行数据:")
print(test_par.head())

# 遍历test_par的每一行
for index, row in test_par.iterrows():
    # 将df每行最大值的cwe写入test_par的new_target列
    max_cwe = df.loc[index].idxmax()  # 获取当前行最大值对应的列名
    test_par.at[index, 'new_target'] = max_cwe  # 将最大值对应的cwe写入new_target列
    if max_cwe != row['level1']:
        print(f"行 {index} 的 level1 值为 {row['level1']}，但 new_target 被设置为 {max_cwe}")

# 保存修改后的DataFrame到新的parquet文件
test_par.to_parquet("/root/autodl-tmp/data/test_cwe_with_new_target.parquet", index=False)

# 打印修改后的DataFrame的前几行
print("修改后的前几行数据:")
print(test_par.head())
