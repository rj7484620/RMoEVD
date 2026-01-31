import os
import json
import argparse
import numpy as np
import pandas as pd
from autogluon.multimodal import MultiModalPredictor


# -------------------------------------------------
# Text cleaning
# -------------------------------------------------
def clean_function_text(func: str) -> str:
    lines = func.split("\n")
    lines = [line.strip() for line in lines if line.strip()]
    return "\n".join(lines)


# -------------------------------------------------
# Relabeling logic (core of RQ3)
# -------------------------------------------------
def relabel_by_new_target(df: pd.DataFrame, positive_new_targets):
    """
    Binary relabeling:
    target = 1 iff original target == 1 AND new_target in positive_new_targets
    """
    df = df.copy()
    mask = (df["target"] == 1) & (df["new_target"].isin(positive_new_targets))
    df["target"] = mask.astype(int)
    print(f"Positive samples after relabeling: {df['target'].sum()}")
    return df


# -------------------------------------------------
# Data loading
# -------------------------------------------------
def load_split(path):
    df = pd.read_parquet(path)
    df["function"] = df["function"].apply(clean_function_text)
    return df


# -------------------------------------------------
# Model path
# -------------------------------------------------
def build_model_path(model_name: str, seed: int, expert_name: str) -> str:
    model_short = model_name.split("/")[-1]
    return f"../models/output_{model_short}_seed{seed}/{expert_name}"


# -------------------------------------------------
# Training / Loading model
# -------------------------------------------------
def load_or_train(train_df, val_df, test_df, args, expert_name):
    model_path = build_model_path(args.model_name, args.seed, expert_name)

    if os.path.exists(model_path):
        predictor = MultiModalPredictor.load(model_path)
        predictor.set_num_gpus(1)
        return predictor

    predictor = MultiModalPredictor(label="target", eval_metric="f1", path=model_path)

    predictor.fit(
        train_data=train_df,
        tuning_data=val_df,
        seed=args.seed,
        hyperparameters={
            "model.hf_text.checkpoint_name": args.model_name,
            "env.precision": "bf16-mixed",
        },
    )

    predictor.set_num_gpus(1)

    eval_result = predictor.evaluate(test_df, metrics=["f1", "average_precision", "precision", "recall"])
    print(eval_result)

    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, "eval_result.json"), "w") as f:
        json.dump(eval_result, f)

    return predictor


# -------------------------------------------------
# Prediction
# -------------------------------------------------
def predict_and_save(predictor, df, model_path, name):
    print(f"Predicting on {name}...")
    proba = predictor.predict_proba(df[["function"]], as_multiclass=False, as_pandas=False)
    np.save(os.path.join(model_path, f"{name}_pred_proba.npy"), proba)


# -------------------------------------------------
# Main
# -------------------------------------------------
def main(args):

    train_df = load_split(args.train_file)
    val_df = load_split(args.val_file)
    test_df = load_split(args.test_file)

    # -------- Mode handling --------
    if args.mode == "single":
        positive_targets = [args.cwe]
        expert_name = args.cwe

    elif args.mode == "merged":
        positive_targets = args.cwe_group.split(",")
        expert_name = args.group_name if args.group_name else "merged_" + "_".join(positive_targets)

    else:  # binary
        positive_targets = None
        expert_name = "binary"

    # Apply relabeling if needed
    if positive_targets:
        train_df = relabel_by_new_target(train_df, positive_targets)
        val_df = relabel_by_new_target(val_df, positive_targets)
        test_df = relabel_by_new_target(test_df, positive_targets)

    # Keep required columns
    train_df = train_df[["function", "target"]]
    val_df = val_df[["function", "target"]]
    test_df = test_df[["function", "target"]]

    print("Dataset preparation complete.")

    predictor = load_or_train(train_df, val_df, test_df, args, expert_name)

    model_path = build_model_path(args.model_name, args.seed, expert_name)

    predict_and_save(predictor, test_df, model_path, "test")
    predict_and_save(predictor, val_df, model_path, "val")

    if args.pred_on_train:
        predict_and_save(predictor, train_df, model_path, "train")


# -------------------------------------------------
# CLI
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="RQ3 Expert Training Pipeline")

    parser.add_argument("--mode", choices=["binary", "single", "merged"], default="single")
    parser.add_argument("--cwe", type=str, default="CWE-189")
    parser.add_argument("--cwe_group", type=str, help="Comma-separated CWE list for merged expert")
    parser.add_argument("--group_name", type=str, help="Name for merged expert directory")

    parser.add_argument("--model_name", type=str, default="microsoft/codebert-base")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--train_file", type=str, default="../data/train_cwe_with_new_target.parquet")
    parser.add_argument("--val_file", type=str, default="../data/val_cwe_with_new_target.parquet")
    parser.add_argument("--test_file", type=str, default="../data/test_cwe_with_new_target.parquet")

    parser.add_argument("--pred_on_train", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
