import os
import pandas as pd


def add_new_target_column(
    pred_path: str,
    label_path: str,
    output_path: str,
    split_name: str,
    label_col: str = "level1",
    new_col: str = "new_target",
    show_head: int = 5,
    max_mismatch_logs: int = 50,
):
    """
    Add a Top-1 predicted label column (`new_target`) into a parquet dataset.

    Steps:
      1) Load prediction probabilities (DataFrame): rows = samples, cols = classes.
      2) Load ground-truth dataset (parquet).
      3) Compute Top-1 predicted class for each row using idxmax across columns.
      4) Write the predicted class into `new_target`.
      5) Print mismatches between ground truth and prediction (optional).
      6) Save the updated dataset to a new parquet file.
    """

    # -----------------------------
    # Load predictions
    # -----------------------------
    pred_df = pd.read_pickle(pred_path)
    print(f"\n[{split_name}] Prediction Data Type: {type(pred_df)}")
    print(f"[{split_name}] Prediction Data Shape: {getattr(pred_df, 'shape', 'no shape')}")
    print(f"[{split_name}] Prediction Preview:")
    print(pred_df.head(show_head))

    # -----------------------------
    # Load dataset labels/features
    # -----------------------------
    data_df = pd.read_parquet(label_path)
    print(f"\n[{split_name}] Dataset Data Type: {type(data_df)}")
    print(f"[{split_name}] Dataset Data Shape: {getattr(data_df, 'shape', 'no shape')}")
    print(f"[{split_name}] Dataset Preview:")
    print(data_df.head(show_head))

    # -----------------------------
    # Safety check: align indices
    # -----------------------------
    # This ensures pred_df and data_df refer to the same samples in the same order.
    if len(pred_df) != len(data_df):
        raise ValueError(
            f"[{split_name}] Length mismatch: pred_df has {len(pred_df)} rows, "
            f"but dataset has {len(data_df)} rows."
        )

    # If both are RangeIndex (0..n-1), alignment is straightforward.
    # Otherwise, enforce positional alignment by resetting indices.
    if not pred_df.index.equals(data_df.index):
        print(f"[{split_name}] Warning: index mismatch detected. Forcing positional alignment via reset_index(drop=True).")
        pred_df = pred_df.reset_index(drop=True)
        data_df = data_df.reset_index(drop=True)

    # -----------------------------
    # Compute Top-1 predicted label per row
    # -----------------------------
    # idxmax(axis=1) returns the column name with maximum value for each row
    data_df[new_col] = pred_df.idxmax(axis=1)

    # -----------------------------
    # Report mismatches (prediction != ground truth)
    # -----------------------------
    if label_col in data_df.columns:
        mismatch_df = data_df[data_df[new_col] != data_df[label_col]]
        print(f"\n[{split_name}] Mismatches: {len(mismatch_df)} / {len(data_df)}")

        # Print a limited number of mismatch examples
        if max_mismatch_logs > 0 and len(mismatch_df) > 0:
            print(f"[{split_name}] Showing up to {max_mismatch_logs} mismatch examples:")
            for i, (idx, row) in enumerate(mismatch_df.iterrows()):
                if i >= max_mismatch_logs:
                    break
                print(f"  Sample {idx}: Ground Truth = {row[label_col]} | Predicted = {row[new_col]}")
    else:
        print(f"[{split_name}] Note: label column '{label_col}' not found. Skipping mismatch report.")

    # -----------------------------
    # Save updated dataset
    # -----------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data_df.to_parquet(output_path, index=False)

    print(f"\n[{split_name}] Updated dataset preview:")
    print(data_df.head(show_head))
    print(f"[{split_name}] Saved to: {output_path}")


if __name__ == "__main__":
    # =========================================================
    # Update these paths to match your project layout.
    # Each split needs:
    #   - a prediction probability .pkl file
    #   - the corresponding dataset .parquet file
    # =========================================================
    splits = {
        "train": {
            "pred_path": "../models/output_codebert-base_seed42/focal_ep20_bs512_eval_f1_macro_gamma1/trainset_pred_proba.pkl",
            "label_path": "../data/train_cwe.parquet",
            "output_path": "../data/train_cwe_with_new_target.parquet",
        },
        "val": {
            "pred_path": "../models/output_codebert-base_seed42/focal_ep20_bs512_eval_f1_macro_gamma1/valset_pred_proba.pkl",
            "label_path": "../data/val_cwe.parquet",
            "output_path": "../data/val_cwe_with_new_target.parquet",
        },
        "test": {
            "pred_path": "../models/output_codebert-base_seed42/focal_ep20_bs512_eval_f1_macro_gamma1/testset_pred_proba.pkl",
            "label_path": "../data/test_cwe.parquet",
            "output_path": "../data/test_cwe_with_new_target.parquet",
        },
    }

    for split_name, paths in splits.items():
        add_new_target_column(
            pred_path=paths["pred_path"],
            label_path=paths["label_path"],
            output_path=paths["output_path"],
            split_name=split_name,
            label_col="level1",
            new_col="new_target",
            show_head=5,
            max_mismatch_logs=50,
        )
