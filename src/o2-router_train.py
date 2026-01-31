import json
from autogluon.core.utils.loaders import load_pd
from datasets import load_dataset
import pandas as pd
from autogluon.multimodal import MultiModalPredictor
import uuid
import torch
from sklearn.model_selection import train_test_split
import sklearn
import numpy as np
import os
from autogluon.core.metrics import make_scorer
import sys
import argparse
from imblearn.over_sampling import RandomOverSampler


def clean_func(func):
    lines = func.split("\n")
    new_lines = []
    for line in lines:
        # trim leading and trailing whitespaces
        line = line.strip()
        # remove empty lines
        if line:
            new_lines.append(line)
    return "\n".join(new_lines)

def trainer(train_pd, val_pd, test_pd, args):
    
    weights = []
    for type in train_pd['level1'].value_counts().keys():
        class_data = train_pd[train_pd.level1 == type]
        weights.append(1 / (class_data.shape[0] / train_pd.shape[0]))
        print(f"class {type}: num samples {len(class_data)}")
    weights = list(np.array(weights) / np.sum(weights))
    weights = [float(w) for w in weights]
    
    print(weights)
    
    selected_model = args.model_name

    epochs = args.epochs
    batch_size = args.batch_size
    gamma = 1
    if args.ros:
        model_path = f"/root/autodl-tmp/output_{selected_model.split('/')[-1]}_seed{args.seed}/ros_ep{epochs}_bs{batch_size}_eval_f1_macro"
    else:
        model_path = f"/root/autodl-tmp/output_{selected_model.split('/')[-1]}_seed{args.seed}/focal_ep{epochs}_bs{batch_size}_eval_f1_macro_gamma{gamma}"
    
    if os.path.exists(model_path):
        # load model
        predictor = MultiModalPredictor.load(model_path)
        predictor.set_num_gpus(1)
        
    else:
        # 1）基础超参
        param_dicts = {
            "model.hf_text.checkpoint_name": selected_model,
            "env.precision": "bf16-mixed",
            "env.per_gpu_batch_size": 32,
            "env.batch_size": batch_size,
            # 注意：这里用 optim.max_epochs 而不是 optimization.max_epochs
            "optim.max_epochs": epochs,
        }
        
        if not args.ros:
            print("use focal loss")
            # focal-loss 相关 key 也都要用 optim 前缀
            param_dicts.update({
                "optim.loss_func": "focal_loss",
                "optim.focal_loss.alpha": weights,    # per-class 权重列表
                "optim.focal_loss.gamma": 1.0,
                "optim.focal_loss.reduction": "sum",
            })
        else:
            print("use ROS")
            ros = RandomOverSampler(random_state=0)
            train_pd, y_resampled = ros.fit_resample(train_pd, train_pd["level1"])
            train_pd = train_pd.reset_index(drop=True)
            print(train_pd.level1.value_counts().to_dict())
        
        predictor = MultiModalPredictor(
            label="level1",
            path=model_path,
            eval_metric="f1_macro"
        )
        predictor.fit(
            train_data=train_pd,
            tuning_data=val_pd,
            hyperparameters=param_dicts
        )


        eval_result = predictor.evaluate(test_pd,metrics = ['f1_macro','f1_micro','f1_weighted','accuracy','mcc'])
        print(eval_result)
        with open(f"{model_path}/multicls_eval_result.json", "w") as f:
            json.dump(eval_result, f)

    
    inference_test = False
    if inference_test:
        test_pd = pd.read_parquet(args.test_file)
        test_pd['function'] = test_pd['function'].apply(clean_func)
        test_dataset = test_pd[["function", "level1"]]
        test_result = predictor.predict_proba(test_dataset[["function"]])
        test_result.to_pickle(f"{model_path}/testset_pred_proba.pkl")


    inference_train = True
    if inference_train:
        train_pd = pd.read_parquet(args.train_file)
        train_pd['function'] = train_pd['function'].apply(clean_func)
        train_dataset = train_pd[["function", "level1"]]
        train_result = predictor.predict_proba(train_dataset[["function"]])
        train_result.to_pickle(f"{model_path}/trainset_pred_proba.pkl")

    inference_val = False
    if inference_val:
        val_pd = pd.read_parquet(args.val_file)
        val_pd['function'] = val_pd['function'].apply(clean_func)
        val_dataset = val_pd[["function", "level1"]]
        val_result = predictor.predict_proba(val_dataset[["function"]])
        val_result.to_pickle(f"{model_path}/valset_pred_proba.pkl")


def parse_args():
    parser = argparse.ArgumentParser(
        description="sequence classification task")
    parser.add_argument(
        "--cwe",
        type=str,
        default="binary",
        help="cwe name",
    )

    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/codebert-base",
        help="model name",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="epochs",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="batch_size",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="ag train seed",
    )
    parser.add_argument(
        "--ros",
        type=bool,
        default=False,
        help="whether to do ros or focalloss",
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="/root/autodl-tmp/data/train_cwe.parquet",
        help="train file",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="/root/autodl-tmp/data/test_cwe.parquet",
        help="test file",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="/root/autodl-tmp/data/val_cwe.parquet",
        help="val file",
    )
    args = parser.parse_args()

    return args


def main():
    train_pd = pd.read_parquet(args.train_file)
    train_pd['function'] = train_pd['function'].apply(clean_func)
    train_pd = train_pd[train_pd['target']==1]
    
    val_pd = pd.read_parquet(args.val_file)
    val_pd['function'] = val_pd['function'].apply(clean_func)
    val_pd = val_pd[val_pd['target']==1]
    
    test_pd = pd.read_parquet(args.test_file)
    test_pd['function'] = test_pd['function'].apply(clean_func)
    test_pd = test_pd[test_pd['target']==1]
    
    def find_cwes(df,level):
        cwe_list = list(df[level].value_counts().keys())
        # remove nan
        cwe_list = [x for x in cwe_list if str(x) != 'nan']
        return cwe_list
    
    lv1_cwes = find_cwes(train_pd,'level1')
    
    train_lv1 = train_pd[train_pd['level1'].isin(lv1_cwes)]
    val_lv1 = val_pd[val_pd['level1'].isin(lv1_cwes)]
    test_lv1 = test_pd[test_pd['level1'].isin(lv1_cwes)]

    
    train_pd = train_lv1[["function", "level1"]]
    val_pd = val_lv1[["function", "level1"]]
    test_pd = test_lv1[["function", "level1"]]
    
    trainer(train_pd, val_pd, test_pd, args)
    
    
if __name__ == "__main__":
    args = parse_args()
    main()