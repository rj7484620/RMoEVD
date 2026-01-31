import os
import json
import argparse
import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor


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
    
    selected_model = args.model_name
    model_path = f"/root/autodl-tmp/output_{selected_model.split('/')[-1]}_seed{args.seed}/{args.cwe}"
    if os.path.exists(model_path):
        # 路径存在则直接加载模型
        predictor = MultiModalPredictor.load(model_path)
        predictor.set_num_gpus(1)
    else:
        predictor = MultiModalPredictor(
            label='target', eval_metric="f1", path=model_path
        )
        predictor.fit(
            train_data=train_pd,
            tuning_data=val_pd,
            seed = args.seed,
            hyperparameters={
                "model.hf_text.checkpoint_name": selected_model,
                "env.precision": "bf16-mixed",
            },
        )
        # 在单卡上推理
        predictor.set_num_gpus(1)
        eval_result = predictor.evaluate(test_pd,metrics = ['f1','average_precision','precision','recall'])
        print(eval_result)
        with open(f"{model_path}/eval_result.json", "w") as f:
            json.dump(eval_result, f)
            
    pred_on_test = True
    if pred_on_test:
        print('predict on test')
        test_result_pd_dir = f"{model_path}/test_result_pd.pkl"
        if os.path.exists(test_result_pd_dir):
            test_dataset = pd.read_pickle(test_result_pd_dir)
        else:
            test_pd = pd.read_parquet(args.test_file)
            test_pd['function'] = test_pd['function'].apply(clean_func)
            test_dataset = test_pd[["function", "target"]]
        pred_proba = predictor.predict_proba(test_dataset[["function"]],as_multiclass =False,as_pandas=False)
        np.save(model_path + "/test_pred_proba.npy",pred_proba)

    pred_on_val = True
    # to obtain optimal threshold on validation set
    if pred_on_val:
        print('predict on val')
        val_result_pd_dir = f"{model_path}/val_result_pd.pkl"
        if os.path.exists(val_result_pd_dir):
            val_dataset = pd.read_pickle(val_result_pd_dir)
        else:
            val_pd = pd.read_parquet(args.val_file)
            val_pd['function'] = val_pd['function'].apply(clean_func)
            val_dataset = val_pd[["function", "target"]]
        pred_proba = predictor.predict_proba(val_dataset[["function"]],as_multiclass =False,as_pandas=False)
        np.save(model_path + "/val_pred_proba.npy",pred_proba)
    
    pred_on_train = False
    # to evaluate the performance of ensemble, otherwise not required
    if pred_on_train:
        print('predict on train')
        train_result_pd_dir = f"{model_path}/train_result_pd.pkl"
        if os.path.exists(train_result_pd_dir):
            train_dataset = pd.read_pickle(train_result_pd_dir)
        else:
            train_pd = pd.read_parquet(args.train_file)
            train_pd['function'] = train_pd['function'].apply(clean_func)
            train_dataset = train_pd[["function", "target"]]
        pred_proba = predictor.predict_proba(train_dataset[["function"]],as_multiclass =False,as_pandas=False)
        np.save.to_numpy(model_path + "/train_pred_proba.npy",pred_proba)

    



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
        "--force_cwe",
        action="store_true",
        help="Whether to consider other cwe as negative label",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="microsoft/codebert-base",
        help="model name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="ag train seed",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="出现此 flag 则启用调试模式"
    )
    parser.add_argument(
        "--train_file",
        type=str,
        default="/root/autodl-tmp/data/train_cwe_with_new_target.parquet",
        help="train file",
    )
    parser.add_argument(
        "--test_file",
        type=str,
        default="/root/autodl-tmp/data/test_cwe_with_new_target.parquet",
        help="test file",
    )
    parser.add_argument(
        "--val_file",
        type=str,
        default="/root/autodl-tmp/data/val_cwe_with_new_target.parquet",
        help="val file",
    )
    args = parser.parse_args()

    return args

def get_cwe_dataset(df,cwe,other_cwe_as_neg=False):

    df_pos = df[df['level1'] == cwe]
    if other_cwe_as_neg:
        df_neg = df[df['level1'] != cwe]
        df_neg['target'] = 0
    else:
        df_neg = df[df['target'] == 0]
    
    df_cwe = pd.concat([df_pos,df_neg],ignore_index=True)
    return df_cwe

def get_cwe_dataset_new(df):
    cnt = 0
    cwe = ['CWE-691', 'CWE-189', 'CWE-703', 'CWE-254']
    print(cwe)
    for index, row in df.iterrows():
        if row['target'] == 1 and row['new_target'] in cwe:
            df.at[index, 'target'] = 1
            cnt += 1
        else:
            df.at[index, 'target'] = 0
    print(f"Total positive samples for {cwe}: {cnt}")
    return df


def main():    
    train_pd = pd.read_parquet(args.train_file)
    train_pd['function'] = train_pd['function'].apply(clean_func)
    val_pd = pd.read_parquet(args.val_file)
    val_pd['function'] = val_pd['function'].apply(clean_func)
    test_pd = pd.read_parquet(args.test_file)
    test_pd['function'] = test_pd['function'].apply(clean_func)
    
    
    if args.cwe != "binary":
        train_pd = get_cwe_dataset_new(train_pd)
        print(f'train_pd_all: {train_pd["target"].value_counts()}')
        
        val_pd = get_cwe_dataset_new(val_pd)
        print(f'val_pd_all: {val_pd["target"].value_counts()}')
        
        test_pd = get_cwe_dataset_new(test_pd)
        print(f'test_pd_all: {test_pd["target"].value_counts()}')
    
    
    train_pd = train_pd[["function", "target"]]
    val_pd = val_pd[["function", "target"]]
    test_pd = test_pd[["function", "target"]]
    
    print('数据准备完毕')
    # trainer(train_pd, val_pd, test_pd, args)
    
    
if __name__ == "__main__":
    args = parse_args()
    main()