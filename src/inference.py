import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from utils import calculate_max_f1_threshold

selected_model = "microsoft/codebert-base"

output_path = f"../models/output_{selected_model.split('/')[-1]}_seed0"
router_path = f"../models/output_{selected_model.split('/')[-1]}_seed42_ori"

test_pd = pd.read_parquet(f"../data/test_cwe.parquet")
train_pd = pd.read_parquet(f"../data/train_cwe.parquet")
val_pd = pd.read_parquet(f"../data/val_cwe.parquet")

cwe_list = [
    "CWE-664",
    "CWE-unknown",
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

df_test = pd.DataFrame(columns=cwe_list)
df_val = pd.DataFrame(columns=cwe_list)
for cwe in cwe_list:
    test_pred_prob = np.load(f"{output_path}/{cwe}/test_pred_proba.npy")
    df_test[cwe] = test_pred_prob

    val_pred_prob = np.load(f"{output_path}/{cwe}/val_pred_proba.npy")
    df_val[cwe] = val_pred_prob

df_test.to_pickle(f"{output_path}/test_pred_prob.pkl")
df_val.to_pickle(f"{output_path}/val_pred_prob.pkl")


valset_result_pd_dir = f"{output_path}/val_pred_prob.pkl"
valset_result_pd = pd.read_pickle(valset_result_pd_dir)
# change according to the path
valset_mtcls_pd_dir = (
    f"{router_path}/focal_ep20_bs512_eval_f1_macro_gamma1/valset_pred_proba.pkl"
)
valset_mtcls_pd = pd.read_pickle(valset_mtcls_pd_dir)
# add _pred_proba to the column name
valset_result_pd.columns = [
    f"{col}_pred_proba" if col != "target" else col for col in valset_result_pd.columns
]

testset_result_pd_dir = f"{output_path}/test_pred_prob.pkl"
testset_result_pd = pd.read_pickle(testset_result_pd_dir)
# change according to the path
testset_mtcls_pd_dir = (
    f"{router_path}/focal_ep20_bs512_eval_f1_macro_gamma1/testset_pred_proba.pkl"
)
testset_mtcls_pd = pd.read_pickle(testset_mtcls_pd_dir)
# add _pred_proba to the column name
testset_result_pd.columns = [
    f"{col}_pred_proba" if col != "target" else col for col in testset_result_pd.columns
]


testset_mtcls_pd.columns = [f"class_{k}_prob" for k in testset_mtcls_pd.columns]
# trainset_mtcls_pd.columns = [f"class_{k}_prob" for k in testset_mtcls_pd.columns]
valset_mtcls_pd.columns = [f"class_{k}_prob" for k in valset_mtcls_pd.columns]

test_pd_levels_info = test_pd[["level1", "target"]]
val_pd_levels_info = val_pd[["level1", "target"]]

testset_result_pd = pd.concat(
    [testset_result_pd, testset_mtcls_pd, test_pd_levels_info], axis=1
)
valset_result_pd = pd.concat(
    [valset_result_pd, valset_mtcls_pd, val_pd_levels_info], axis=1
)


# import softmax


def find_top_2_prob_and_do_softmax(row, topn=2):

    # ...existing code...
    level_1_idx = {cwe: idx for idx, cwe in enumerate(cwe_list)}
    # ...existing code...

    included_rows = row[[f"class_{k}_prob" for k in level_1_idx.keys()]]
    # convert value to np array
    included_rows = included_rows.to_numpy()
    # print(included_rows)
    # find top 2 prob
    top_2_prob_idx = included_rows.argsort()[-topn:]

    # new array that only contains top 2 prob
    top_2_prob = included_rows[top_2_prob_idx]
    # print(top_2_prob)
    # to float
    top_2_prob = top_2_prob.astype(float)
    # softmax
    top_2_prob = softmax(top_2_prob)
    # return included_rows
    # idx：prob dict
    idx_prob_dict = {idx: prob for idx, prob in zip(top_2_prob_idx, top_2_prob)}
    # for each idx and prob, get class name by idx
    idx_prob_dict = {list(level_1_idx.keys())[k]: v for k, v in idx_prob_dict.items()}
    # for prob in idx_prob_dict, multiple value with column name idx
    idx_prob_dict = {k: v * row[f"{k}_pred_proba"] for k, v in idx_prob_dict.items()}
    # do sum on value of all value
    sum_prob = sum(idx_prob_dict.values())
    # print(idx_prob_dict)
    return sum_prob


# find best threshold with validation set
valset_result_pd["top2_softmax_prob"] = valset_result_pd.apply(
    lambda row: find_top_2_prob_and_do_softmax(row, topn=2), axis=1
)

best_threshold, f1, precision, recall, accuracy = calculate_max_f1_threshold(
    valset_result_pd, col="top2_softmax_prob"
)
print(f"Best threshold: {best_threshold}")
valset_result_pd["top2_softmax_pred"] = valset_result_pd["top2_softmax_prob"].apply(
    lambda x: 1 if x >= best_threshold else 0
)
print(
    classification_report(
        valset_result_pd["target"], valset_result_pd["top2_softmax_pred"], digits=3
    )
)


testset_result_pd["top2_softmax_prob"] = testset_result_pd.apply(
    lambda row: find_top_2_prob_and_do_softmax(row, topn=2), axis=1
)

testset_result_pd["top2_softmax_pred"] = testset_result_pd["top2_softmax_prob"].apply(
    lambda x: 1 if x >= best_threshold else 0
)
print(
    classification_report(
        testset_result_pd["target"], testset_result_pd["top2_softmax_pred"], digits=3
    )
)
