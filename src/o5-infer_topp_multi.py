import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score

def calculate_max_f1_threshold(df, col='pred_prob'):
    thresholds = np.linspace(0, 1, 100)
    max_f1 = 0
    best_threshold = 0
    max_precision = 0
    max_recall = 0
    max_accuracy = 0
    
    for threshold in thresholds:
        predictions = df[col].apply(lambda x: 1 if x >= threshold else 0)
        f1 = f1_score(df['target'], predictions)
        
        if f1 > max_f1:
            max_f1 = f1
            best_threshold = threshold
    
    predictions = df[col].apply(lambda x: 1 if x >= best_threshold else 0)       
    f1 = f1_score(df['target'], predictions)        
    precision = precision_score(df['target'], predictions)
    recall = recall_score(df['target'], predictions)
    accuracy = accuracy_score(df['target'], predictions)
    
    return best_threshold, f1, precision, recall, accuracy

# === 1. 配置模型名称和路径 ===
selected_model = "microsoft/codebert-base"
# 主模型输出目录（seed=0）
output_path = f"/root/autodl-tmp/output_{selected_model.split('/')[-1]}_seed0_ori"
# 路由模型输出目录（seed=42）
router_path = f"/root/autodl-tmp/output_{selected_model.split('/')[-1]}_seed42_ori"

# === 2. 定义要处理的 CWE 类别列表 ===
cwe_list = [
    "CWE-664", "CWE-unknown", "CWE-707", "CWE-399",
    "CWE-264", "CWE-691", "CWE-284", "CWE-682",
    "CWE-189", "CWE-other", "CWE-703", "CWE-254",
]
# 构建从索引到 CWE 代码的映射，以保持顺序一致
idx_to_cwe = {idx: cwe for idx, cwe in enumerate(cwe_list)}

# === 3. 读取原始数据集标签信息 ===
train_df = pd.read_parquet("/root/autodl-tmp/data/train_cwe.parquet")
val_df   = pd.read_parquet("/root/autodl-tmp/data/val_cwe.parquet")
test_df  = pd.read_parquet("/root/autodl-tmp/data/test_cwe.parquet")

# === 4. 加载二分类模型预测概率到 DataFrame ===
def load_binary_probs(path, cwe_list, split: str) -> pd.DataFrame:
    """
    从指定目录中加载每个 CWE 的二分类预测概率（.npy），
    并返回一个样本数 × 类别数的 DataFrame，列名为 {CWE}_pred_proba。
    """
    df = pd.DataFrame()
    for cwe in cwe_list:
        probs = np.load(f"{path}/{cwe}/{split}_pred_proba.npy")
        df[f"{cwe}_pred_proba"] = probs
    return df

binary_val_df = load_binary_probs(output_path, cwe_list, split="val")
binary_test_df = load_binary_probs(output_path, cwe_list, split="test")

# === 5. 加载多分类模型预测概率并重命名列 ===
def load_multiclass_probs(path, folder: str, split: str) -> pd.DataFrame:
    """
    从路由模型目录加载多分类预测概率，并将列名统一为 class_{CWE}_prob。
    """
    df = pd.read_pickle(f"{path}/{folder}/{split}set_pred_proba.pkl")
    # 原始列名应为 CWE 代码，与 cwe_list 顺序一致
    df.columns = [f"class_{col}_prob" for col in df.columns]
    return df

# 根据实际文件夹名替换
folder_name = "focal_ep20_bs512_eval_f1_macro_gamma1"
mtcls_val_df  = load_multiclass_probs(router_path, folder_name, split="val")
mtcls_test_df = load_multiclass_probs(router_path, folder_name, split="test")

# === 6. 合并所有预测概率与原始标签信息 ===
valset = pd.concat([
    binary_val_df,
    mtcls_val_df,
    val_df[['level1', 'target']]
], axis=1)

testset = pd.concat([
    binary_test_df,
    mtcls_test_df,
    test_df[['level1', 'target']]
], axis=1)

# ...existing code...

def compute_topn_softmax_score(row: pd.Series, topn: int) -> float:
    """
    选取 topn 个概率最大的专家做 softmax 融合。
    """
    multiclass_probs = np.array([
        row[f"class_{cwe}_prob"] for cwe in cwe_list
    ], dtype=float)
    top_indices = np.argsort(multiclass_probs)[-topn:][::-1]
    top_n_probs = softmax(multiclass_probs[top_indices])
    fused_score = 0.0
    for rel_prob, idx in zip(top_n_probs, top_indices):
        cwe_code = idx_to_cwe[idx]
        binary_prob = row[f"{cwe_code}_pred_proba"]
        fused_score += rel_prob * binary_prob
    return fused_score

# 计算每种专家数量下的最优阈值
topn_thr_dict = {}
topn_f1_dict = {}
for topn in range(1, len(cwe_list)+1):
    valset[f'top{topn}_softmax_prob'] = valset.apply(lambda row: compute_topn_softmax_score(row, topn), axis=1)
    best_thr, best_f1, best_prec, best_rec, best_acc = calculate_max_f1_threshold(
        valset, col=f'top{topn}_softmax_prob'
    )
    topn_thr_dict[topn] = best_thr
    topn_f1_dict[topn] = best_f1
    print(f"top{topn}: 最优阈值={best_thr:.4f}, F1={best_f1:.4f}, 精度={best_prec:.4f}, 召回={best_rec:.4f}, 准确率={best_acc:.4f}")

# topp融合函数，返回实际选取专家数
def compute_topp_softmax_score_with_n(row: pd.Series, topp: float = 0.8, min_expert_num: int = 2):
    multiclass_probs = np.array([
        row[f"class_{cwe}_prob"] for cwe in cwe_list
    ], dtype=float)
    sorted_indices = np.argsort(multiclass_probs)[::-1]
    sorted_probs = multiclass_probs[sorted_indices]
    cum_probs = np.cumsum(sorted_probs)
    total_prob = cum_probs[-1]
    cum_probs_norm = cum_probs / total_prob
    selected_count = np.searchsorted(cum_probs_norm, topp) + 1
    # 保证至少选min_expert_num个专家
    selected_count = max(selected_count, min_expert_num)
    top_indices = sorted_indices[:selected_count]
    top_n_probs = softmax(multiclass_probs[top_indices])
    fused_score = 0.0
    for rel_prob, idx in zip(top_n_probs, top_indices):
        cwe_code = idx_to_cwe[idx]
        binary_prob = row[f"{cwe_code}_pred_proba"]
        fused_score += rel_prob * binary_prob
    #print(f"选取专家数: {selected_count}, 累计概率: {cum_probs_norm[selected_count-1]:.4f}")
    return fused_score, selected_count

# === 8. 在验证集上计算topp融合分数并选阈值（仅演示，实际阈值已在上面计算） ===
valset['topp_softmax_prob'], valset['topp_expert_num'] = zip(*valset.apply(compute_topp_softmax_score_with_n, axis=1))

# 根据实际专家数选用对应阈值进行预测
valset['topp_softmax_pred'] = [
    int(prob >= topn_thr_dict[num]) for prob, num in zip(valset['topp_softmax_prob'], valset['topp_expert_num'])
]
print("验证集评估报告（topp动态阈值）：")
print(classification_report(valset['target'], valset['topp_softmax_pred'], digits=3))

# === 9. 在测试集上应用同样策略 ===
testset['topp_softmax_prob'], testset['topp_expert_num'] = zip(*testset.apply(compute_topp_softmax_score_with_n, axis=1))
testset['topp_softmax_pred'] = [
    int(prob >= topn_thr_dict[num]) for prob, num in zip(testset['topp_softmax_prob'], testset['topp_expert_num'])
]
print("测试集评估报告（topp动态阈值）：")
print(classification_report(testset['target'], testset['topp_softmax_pred'], digits=3))
# ...existing code...


