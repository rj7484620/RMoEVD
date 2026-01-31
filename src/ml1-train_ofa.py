# -*- coding: utf-8 -*-
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,4,5"

import argparse
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import recall_score
from autogluon.multimodal import MultiModalPredictor

warnings.filterwarnings("ignore")

# ----------------------------- 数据预处理工具函数 ----------------------------- #

def clean_func(func: str) -> str:
    """简单去掉首尾空格 / 空行，保持与原脚本一致"""
    lines = [ln.strip() for ln in func.split("\n") if ln.strip()]
    return "\n".join(lines)

def add_sample_weight(df: pd.DataFrame, pos_weight: float) -> pd.DataFrame:
    """在 DataFrame 中新增 'weight' 列——正例=pos_weight，负例=1"""
    df = df.copy()
    df["weight"] = df["target"].apply(lambda x: pos_weight if x == 1 else 1)
    return df

# ----------------------------- 主训练流程 ----------------------------- #

def trainer(train_pd: pd.DataFrame, val_pd: pd.DataFrame, test_pd: pd.DataFrame, args):
    
    selected_model = args.model_name
    model_path = f"../output_{selected_model.split('/')[-1]}_seed{args.seed}/{args.cwe}"

    if os.path.exists(model_path):
        predictor = MultiModalPredictor.load(model_path)
        predictor.set_num_gpus(1)
        print("检测到已训练模型，直接加载完成！")
        # 在测试集上计算f1,precision,recall,accuracy并写入eval_result.json
        eval_result = predictor.evaluate(test_pd,metrics = ['f1','average_precision','precision','recall'])
        print(eval_result)
        with open(f"{model_path}/eval_result.json", "w") as f:
            json.dump(eval_result, f)

        # 保存最优阈值（在验证集上逼近 target_recall）
        save_best_threshold(predictor, val_pd, args.target_recall, model_path)
    else:
        os.makedirs(model_path, exist_ok=True)
        print("🔹 开始训练 ...")
        predictor = MultiModalPredictor(
            label='target', eval_metric="f1", path=model_path
        )
        predictor.fit(
            train_data=train_pd,
            tuning_data=val_pd,
            seed=args.seed,
            hyperparameters={
                "model.hf_text.checkpoint_name": selected_model,
                "env.precision": "bf16-mixed", 
                "optim.loss_func": "focal_loss",
                "optim.focal_loss.gamma": 2.0, 
                "optim.focal_loss.alpha": [0.058, 0.942],
            },
        )
        predictor.set_num_gpus(1)
        # 在测试集上计算f1,precision,recall,accuracy并写入eval_result.json
        eval_result = predictor.evaluate(test_pd,metrics = ['f1','average_precision','precision','recall'])
        print(eval_result)
        with open(f"{model_path}/eval_result.json", "w") as f:
            json.dump(eval_result, f)

        # 保存最优阈值（在验证集上逼近 target_recall）
        save_best_threshold(predictor, val_pd, args.target_recall, model_path)

    # ---------------------------- 推理阶段 ---------------------------- #
    print("\n>>> 在验证集上推理并保存概率 ...")
    val_pred_proba = predictor.predict_proba(val_pd[["function"]], as_multiclass=False, as_pandas=False)
    np.save(os.path.join(model_path, "val_pred_proba.npy"), val_pred_proba)
    print("\n>>> 在测试集上推理并保存概率 ...")
    test_pred_proba = predictor.predict_proba(test_pd[["function"]], as_multiclass=False, as_pandas=False)
    np.save(os.path.join(model_path, "test_pred_proba.npy"), test_pred_proba)

    # 根据阈值得到最终标签（可选）
    th_path = os.path.join(model_path, "best_threshold.txt")
    if os.path.exists(th_path):
        best_th = float(open(th_path).read().strip())
        test_pred_label = (test_pred_proba >= best_th).astype(int)
        np.save(os.path.join(model_path, "test_pred_label.npy"), test_pred_label)
        print(f"已应用 best_th={best_th:.4f} 生成 test_pred_label.npy")


# ---------------------------------------------------------------------------- #
#                               阈 值 处 理                                     #
# ---------------------------------------------------------------------------- #

def save_best_threshold(predictor: MultiModalPredictor, val_pd: pd.DataFrame, target_recall: float, model_path: str):

    print("\n>>> 阈值扫描以满足目标召回率 ...")
    proba = predictor.predict_proba(val_pd[["function"]], as_multiclass=False, as_pandas=False)
    y_true = val_pd["target"].values

    thresholds = np.linspace(0.5, 0.0, 501)  # 0~0.5 步长0.001
    best_th = 0.5
    for th in thresholds:
        recall = recall_score(y_true, proba >= th)
        if recall >= target_recall:
            best_th = th
            break

    with open(os.path.join(model_path, "best_threshold.txt"), "w") as f:
        f.write(str(best_th))

    print(f"最佳阈值 = {best_th:.4f} (满足 recall ≥ {target_recall})\n")


# ----------------------------- CLI & 主入口 ----------------------------- #

def parse_args():
    parser = argparse.ArgumentParser(description="CodeBERT 二分类高召回训练脚本")
    parser.add_argument("--cwe", type=str, default="binary", help="cwe name")
    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base", help="预训练模型名称")
    parser.add_argument("--seed", type=int, default=0, help="随机种子")
    parser.add_argument("--train_file", type=str, default="../dataset-MoE/train_cwe.parquet")
    parser.add_argument("--test_file", type=str, default="../dataset-MoE/test_cwe.parquet")
    parser.add_argument("--val_file", type=str, default="../dataset-MoE/val_cwe.parquet")

    parser.add_argument("--pos_weight", type=float, default=1.0, help="正例样本权重 (loss 加权)")
    parser.add_argument("--target_recall", type=float, default=0.95, help="阈值搜索目标召回率")

    return parser.parse_args()

def main():
    args = parse_args()

    train_pd = pd.read_parquet(args.train_file)
    val_pd = pd.read_parquet(args.val_file)
    test_pd = pd.read_parquet(args.test_file)

    for df in (train_pd, val_pd, test_pd):
        df["function"] = df["function"].apply(clean_func)

    # 添加样本权重
    # train_pd = add_sample_weight(train_pd[["function", "target"]], args.pos_weight)
    # val_pd = add_sample_weight(val_pd[["function", "target"]], args.pos_weight)
    # test_pd = add_sample_weight(test_pd[["function", "target"]], args.pos_weight)
    # trainer(train_pd, val_pd, test_pd, args)

    # 只用focal loss
    trainer(train_pd[["function", "target"]], val_pd[["function", "target"]], test_pd[["function", "target"]], args)


if __name__ == "__main__":
    main()
