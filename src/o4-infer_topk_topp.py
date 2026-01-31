import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import classification_report
from utils import calculate_max_f1_threshold

# === 1. 配置模型名称和路径 ===
selected_model = "microsoft/codebert-base"
output_path = f"../output_{selected_model.split('/')[-1]}_seed0"
router_path = f"../output_{selected_model.split('/')[-1]}_seed42_ori"

# === 2. 定义要处理的 CWE 类别列表 ===
cwe_list = [
    "CWE-664", "CWE-unknown", "CWE-707", "CWE-399",
    "CWE-264", "CWE-691", "CWE-284", "CWE-682",
    "CWE-189", "CWE-other", "CWE-703", "CWE-254",
]
idx_to_cwe = {idx: cwe for idx, cwe in enumerate(cwe_list)}

# === 3. 读取原始数据集标签信息 ===
train_df = pd.read_parquet("../dataset-MoE/train_cwe.parquet")
val_df   = pd.read_parquet("../dataset-MoE/val_cwe.parquet")
test_df  = pd.read_parquet("../dataset-MoE/test_cwe.parquet")

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

# === 7. 定义 Top-k Softmax 融合函数 ===
def compute_topk_softmax_score(row: pd.Series, topk: int = 2) -> float:
    """
    对多分类模型的 class_{CWE}_prob 概率中选出 topn 最大的两项，
    先做 softmax 归一化，再与对应的二分类预测概率相乘后求和。
    返回最终的融合分数。
    """
    # 1) 用 cwe_list 顺序提取多分类概率
    multiclass_probs = np.array([
        row[f"class_{cwe}_prob"] for cwe in cwe_list
    ], dtype=float)

    # 2) 找到概率最大的 topn 索引
    top_indices = np.argsort(multiclass_probs)[-topk:]

    # 3) 对这 topn 概率做 softmax 归一化
    top_n_probs = softmax(multiclass_probs[top_indices])


    # 4) 加权：softmax 后概率 * 对应二分类概率，并求和
    fused_score = 0.0
    max_prob = 0.0
    for rel_prob, idx in zip(top_n_probs, top_indices):
        cwe_code = idx_to_cwe[idx]
        binary_prob = row[f"{cwe_code}_pred_proba"]
        max_prob = max(max_prob, binary_prob)
        fused_score += rel_prob * binary_prob
    # print(f"fused_score: {fused_score:.4f}, max_prob: {max_prob:.4f}")
    return fused_score, max_prob

def compute_topp_softmax_score(row: pd.Series, topp: float = 0.8, min_expert_num: int = 2) -> float:
    """
    对多分类模型的 class_{CWE}_prob 概率采用 top-p 策略（累计概率达到 topp），
    至少选 min_expert_num 个专家，softmax归一化后与对应二分类概率相乘求和。
    返回最终的融合分数。
    """
    multiclass_probs = np.array([
        row[f"class_{cwe}_prob"] for cwe in cwe_list
    ], dtype=float)
    sorted_indices = np.argsort(multiclass_probs)[::-1]
    sorted_probs = multiclass_probs[sorted_indices]
    cum_probs = np.cumsum(sorted_probs)
    total_prob = cum_probs[-1]
    cum_probs_norm = cum_probs / total_prob
    selected_count = np.searchsorted(cum_probs_norm, topp) + 1
    selected_count = max(selected_count, min_expert_num)
    top_indices = sorted_indices[:selected_count]
    top_n_probs = softmax(multiclass_probs[top_indices])
    max_prob = 0.0
    fused_score = 0.0
    for rel_prob, idx in zip(top_n_probs, top_indices):
        cwe_code = idx_to_cwe[idx]
        binary_prob = row[f"{cwe_code}_pred_proba"]
        max_prob = max(max_prob, binary_prob)
        fused_score += rel_prob * binary_prob
    return fused_score, max_prob

def run_eval(dynamic_threshold=False, use_topk=True, topk=2, topp=0.8, min_expert_num=2):
    if not dynamic_threshold:
        if use_topk:
            testset['top2_softmax_prob'], testset['top2_max_prob'] = zip(*testset.apply(lambda row: compute_topk_softmax_score(row, topk=topk), axis=1))
        else:
            testset['top2_softmax_prob'], testset['top2_max_prob'] = zip(*testset.apply(lambda row: compute_topp_softmax_score(row, topp=topp, min_expert_num=min_expert_num), axis=1))
        testset['top2_softmax_pred_new'] = (testset['top2_max_prob'] >= 0.5).astype(int)
        print(f"测试集评估报告 动态阈值:{dynamic_threshold} Top-k:{topk}：Top-p={topp} Min Expert Num={min_expert_num}")
        print(classification_report(testset['target'], testset['top2_softmax_pred_new'], digits=3))
    else:
        if use_topk:
            valset['top2_softmax_prob'], valset['top2_max_prob'] = zip(*valset.apply(lambda row: compute_topk_softmax_score(row, topk=topk), axis=1))
        else:
            valset['top2_softmax_prob'], valset['top2_max_prob'] = zip(*valset.apply(lambda row: compute_topp_softmax_score(row, topp=topp, min_expert_num=min_expert_num), axis=1))
        best_thr, best_f1, best_prec, best_rec, best_acc = calculate_max_f1_threshold(
            valset, col='top2_softmax_prob'
        )
        print(f"最优阈值: {best_thr}")
        valset['top2_softmax_pred_new'] = (valset['top2_max_prob'] >= best_thr).astype(int)
        print(f"验证集评估报告 动态阈值:{dynamic_threshold} Top-k:{topk}：Top-p={topp} Min Expert Num={min_expert_num}")
        print(classification_report(valset['target'], valset['top2_softmax_pred_new'], digits=3))
        if use_topk:
            testset['top2_softmax_prob'], testset['top2_max_prob'] = zip(*testset.apply(lambda row: compute_topk_softmax_score(row, topk=topk), axis=1))
        else:
            testset['top2_softmax_prob'], testset['top2_max_prob'] = zip(*testset.apply(lambda row: compute_topp_softmax_score(row, topp=topp, min_expert_num=min_expert_num), axis=1))
        testset['top2_softmax_pred_new'] = (testset['top2_max_prob'] >= best_thr).astype(int)
        print(f"测试集评估报告 动态阈值:{dynamic_threshold} Top-k:{topk}：Top-p={topp} Min Expert Num={min_expert_num}")
        print(classification_report(testset['target'], testset['top2_softmax_pred_new'], digits=3))

run_eval(dynamic_threshold=False, use_topk=True, topk=1, topp=0.8, min_expert_num=1)
run_eval(dynamic_threshold=False, use_topk=True, topk=2, topp=0.8, min_expert_num=2)
run_eval(dynamic_threshold=False, use_topk=True, topk=3, topp=0.8, min_expert_num=3)
run_eval(dynamic_threshold=False, use_topk=True, topk=4, topp=0.8, min_expert_num=4)
run_eval(dynamic_threshold=False, use_topk=True, topk=5, topp=0.8, min_expert_num=5)
run_eval(dynamic_threshold=False, use_topk=True, topk=6, topp=0.8, min_expert_num=6)
run_eval(dynamic_threshold=False, use_topk=True, topk=7, topp=0.8, min_expert_num=7)
run_eval(dynamic_threshold=False, use_topk=True, topk=8, topp=0.8, min_expert_num=8)
run_eval(dynamic_threshold=False, use_topk=True, topk=9, topp=0.8, min_expert_num=9)
run_eval(dynamic_threshold=False, use_topk=True, topk=10, topp=0.8, min_expert_num=10)
run_eval(dynamic_threshold=False, use_topk=True, topk=11, topp=0.8, min_expert_num=11)
run_eval(dynamic_threshold=False, use_topk=True, topk=12, topp=0.8, min_expert_num=12)
run_eval(dynamic_threshold=False, use_topk=False, topk=1, topp=0.8, min_expert_num=1)
run_eval(dynamic_threshold=False, use_topk=False, topk=1, topp=0.8, min_expert_num=2)
run_eval(dynamic_threshold=False, use_topk=False, topk=1, topp=0.8, min_expert_num=3)
run_eval(dynamic_threshold=False, use_topk=False, topk=1, topp=0.8, min_expert_num=4)
run_eval(dynamic_threshold=False, use_topk=False, topk=1, topp=0.8, min_expert_num=5)
run_eval(dynamic_threshold=False, use_topk=False, topk=1, topp=0.8, min_expert_num=6)
run_eval(dynamic_threshold=False, use_topk=False, topk=1, topp=0.8, min_expert_num=7)
run_eval(dynamic_threshold=False, use_topk=False, topk=1, topp=0.8, min_expert_num=8)
run_eval(dynamic_threshold=False, use_topk=False, topk=1, topp=0.8, min_expert_num=9)
run_eval(dynamic_threshold=False, use_topk=False, topk=1, topp=0.8, min_expert_num=10)
run_eval(dynamic_threshold=False, use_topk=False, topk=1, topp=0.8, min_expert_num=11)
run_eval(dynamic_threshold=False, use_topk=False, topk=1, topp=0.8, min_expert_num=12)


