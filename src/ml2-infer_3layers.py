# -*- coding: utf-8 -*-
"""
评估脚本（全球二分类 ➜ 路由多分类 ➜ 专家二分类）
================================================
* **Stage‑1**  Global Binary  (高召回)：判定样本是否 *可能* 含漏洞。
* **Stage‑2**  Router Multi‑Class：给出各 CWE 概率，用来挑选若干专家。
* **Stage‑3**  Expert Binary：每个 CWE 一个二分类模型，输出该 CWE 的置信度。

阈值：
-------
* `BIN_THR`: 可固定一个较低阈值（如 0.1）确保高召回
* `EXP_THR`: 同原脚本（默认 0.5 或在验证集动态搜索）
"""

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import classification_report

# ================= 1. 路径与模型名称 =================
selected_model = "microsoft/codebert-base"
global_binary_path = f"../output_{selected_model.split('/')[-1]}_seed0/binary"
expert_prefix_path = f"../output_{selected_model.split('/')[-1]}_seed0_ori"
router_path = f"../output_{selected_model.split('/')[-1]}_seed42_ori"

# =============== 2. CWE 列表 & 映射 =================
cwe_list = [
    "CWE-664", "CWE-unknown", "CWE-707", "CWE-399",
    "CWE-264", "CWE-691", "CWE-284", "CWE-682",
    "CWE-189", "CWE-other", "CWE-703", "CWE-254",
]
idx_to_cwe = {idx: cwe for idx, cwe in enumerate(cwe_list)}

# =============== 3. 加载数据集标签 ====================
train_df = pd.read_parquet("../dataset-MoE/train_cwe.parquet")
val_df   = pd.read_parquet("../dataset-MoE/val_cwe.parquet")
test_df  = pd.read_parquet("../dataset-MoE/test_cwe.parquet")

# =============== 4. 辅助函数 ==========================

def load_probs_npy(path: str, split: str) -> np.ndarray:
    """统一封装 npy 读取，返回 1D ndarray"""
    return np.load(f"{path}/{split}_pred_proba.npy")


def load_global_binary_probs(split: str) -> pd.Series:
    """读取高召回 Global Binary 预测概率，返回 pd.Series"""
    probs = load_probs_npy(global_binary_path, split)
    return pd.Series(probs, name="binary_prob")


def load_expert_binary_probs(split: str) -> pd.DataFrame:
    """读取每个 CWE 专家的二分类概率，列名格式 {CWE}_pred_proba"""
    df = pd.DataFrame()
    for cwe in cwe_list:
        probs = load_probs_npy(f"{expert_prefix_path}/{cwe}", split)
        df[f"{cwe}_pred_proba"] = probs
    return df


def load_router_multiclass_probs(folder: str, split: str) -> pd.DataFrame:
    """读取 Router 多分类概率，列名改为 class_{CWE}_prob"""
    df = pd.read_pickle(f"{router_path}/{folder}/{split}set_pred_proba.pkl")
    df.columns = [f"class_{col}_prob" for col in df.columns]
    return df

# Router 模型子文件夹
router_folder = "focal_ep20_bs512_eval_f1_macro_gamma1"

# =============== 5. 构造 Val / Test DataFrame =========

def build_dataset(split: str) -> pd.DataFrame:
    """组合 GlobalBinary + ExpertBinary + RouterMulti + 真值标签"""
    if split == "val":
        truth_df = val_df
    elif split == "test":
        truth_df = test_df
    else:
        raise ValueError("split must be val or test")

    df = pd.concat([
        load_global_binary_probs(split),                                  # binary_prob
        load_expert_binary_probs(split),                                  # expert probs
        load_router_multiclass_probs(router_folder, split),               # router probs
        truth_df[['level1', 'target']],                                   # ground truth
    ], axis=1)
    return df

# valset = build_dataset("val")
testset = build_dataset("test")

# =============== 6. 融合函数 ==========================

def compute_topk_softmax_score(row: pd.Series, topk: int = 2):
    """与原脚本相同，返回 (fused_score, expert_max_prob)"""
    multiclass_probs = np.array([row[f"class_{cwe}_prob"] for cwe in cwe_list], dtype=float)
    top_indices = np.argsort(multiclass_probs)[-topk:]
    top_n_probs = softmax(multiclass_probs[top_indices])
    fused_score = 0.0
    max_prob = 0.0
    for rel_prob, idx in zip(top_n_probs, top_indices):
        cwe_code = idx_to_cwe[idx]
        binary_prob = row[f"{cwe_code}_pred_proba"]
        max_prob = max(max_prob, binary_prob)
        fused_score += rel_prob * binary_prob
    return fused_score, max_prob


# =============== 7. 评估 =============================

def run_eval(
    binary_thr: float = 0.1,
    expert_thr: float | None = 0.5,
    dynamic_thr: bool = False,
    topk: int = 2,
):
    """综合三层模型输出给出最终报告"""

    # ---------- 阶段 1: Router+Expert 生成 softmax 融合分数 ----------
    print('phase 1')
    # valset[['fused_prob', 'expert_max_prob']] = valset.apply(
    #     lambda r: pd.Series(compute_topk_softmax_score(r, topk=topk)), axis=1
    # )
    testset[['fused_prob', 'expert_max_prob']] = testset.apply(
        lambda r: pd.Series(compute_topk_softmax_score(r, topk=topk)), axis=1
    )

    # ---------- 阶段 2: 阈值确定 ----------
    print('phase 2')
    if dynamic_thr:
        # 只调 expert_thr，binary_thr 通常固定较低以保证召回
        best_thr, *_ = calculate_max_f1_threshold(valset, col='expert_max_prob')
        expert_thr = best_thr
        print(f"[动态阈值] 自动找到 expert_thr = {expert_thr:.4f}")
    else:
        assert expert_thr is not None, "expert_thr must be provided when dynamic_thr=False"

    # ---------- 阶段 3: 最终预测 ----------
    print('phase 3')
    def final_pred(row):
        return int((row['binary_prob'] >= binary_thr) and (row['expert_max_prob'] >= expert_thr))

    # valset['final_pred'] = valset.apply(final_pred, axis=1)
    testset['final_pred'] = testset.apply(final_pred, axis=1)

    # ---------- 阶段 4: 报告 ----------
    print('phase 4')
    # print("========  验证集  ========")
    # print(classification_report(valset['target'], valset['final_pred'], digits=3))
    print("========  测试集  =========")
    print(classification_report(testset['target'], testset['final_pred'], digits=3))


if __name__ == "__main__":
    run_eval(binary_thr=0.016, expert_thr=0.5, dynamic_thr=False, topk=2)

