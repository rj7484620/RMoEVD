import argparse
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import classification_report

from utils import calculate_max_f1_threshold


# -----------------------------
# CWE configuration
# -----------------------------
CWE_LIST = [
    "CWE-664", "CWE-unknown", "CWE-707", "CWE-399",
    "CWE-264", "CWE-691", "CWE-284", "CWE-682",
    "CWE-189", "CWE-other", "CWE-703", "CWE-254",
]
IDX_TO_CWE = {i: cwe for i, cwe in enumerate(CWE_LIST)}


# -----------------------------
# Loading helpers
# -----------------------------
def load_binary_probs(expert_root: str, cwe_list, split: str) -> pd.DataFrame:
    """
    Load expert binary prediction probabilities from .npy files.

    Returns a DataFrame with shape [num_samples, num_experts],
    columns named as: "{CWE}_pred_proba".
    """
    out = {}
    for cwe in cwe_list:
        prob_path = f"{expert_root}/{cwe}/{split}_pred_proba.npy"
        out[f"{cwe}_pred_proba"] = np.load(prob_path)
    return pd.DataFrame(out)


def load_router_probs(router_root: str, folder: str, split: str) -> pd.DataFrame:
    """
    Load router multi-class probabilities from .pkl file.

    The original columns should correspond to CWE codes.
    We rename them to: "class_{CWE}_prob".
    """
    df = pd.read_pickle(f"{router_root}/{folder}/{split}set_pred_proba.pkl")
    df.columns = [f"class_{col}_prob" for col in df.columns]
    return df


# -----------------------------
# Fusion functions
# -----------------------------
def compute_topk_softmax_score(row: pd.Series, topk: int = 2):
    """
    Select Top-k experts by router probabilities, softmax-normalize the selected
    router probs as weights, and compute weighted sum of expert binary probs.

    Returns:
      fused_score: weighted mixture score
      max_prob: max expert binary prob among selected experts (OR-style proxy)
    """
    router_probs = np.array([row[f"class_{cwe}_prob"] for cwe in CWE_LIST], dtype=float)
    top_indices = np.argsort(router_probs)[-topk:]
    weights = softmax(router_probs[top_indices])

    fused_score = 0.0
    max_prob = 0.0
    for w, idx in zip(weights, top_indices):
        cwe_code = IDX_TO_CWE[idx]
        expert_prob = float(row[f"{cwe_code}_pred_proba"])
        max_prob = max(max_prob, expert_prob)
        fused_score += w * expert_prob

    return fused_score, max_prob


def compute_topp_softmax_score(row: pd.Series, topp: float = 0.8, min_expert_num: int = 2):
    """
    Select a variable-size expert set using Top-p (cumulative probability mass) routing.
    Ensure at least 'min_expert_num' experts are selected.

    Returns:
      fused_score: weighted mixture score
      max_prob: max expert binary prob among selected experts (OR-style proxy)
    """
    router_probs = np.array([row[f"class_{cwe}_prob"] for cwe in CWE_LIST], dtype=float)
    sorted_indices = np.argsort(router_probs)[::-1]
    sorted_probs = router_probs[sorted_indices]

    # Normalize cumulative mass to 1 (even if router outputs are not perfectly normalized)
    cum_probs = np.cumsum(sorted_probs)
    total_prob = cum_probs[-1] if cum_probs[-1] != 0 else 1.0
    cum_mass = cum_probs / total_prob

    selected_count = np.searchsorted(cum_mass, topp) + 1
    selected_count = max(selected_count, min_expert_num)

    top_indices = sorted_indices[:selected_count]
    weights = softmax(router_probs[top_indices])

    fused_score = 0.0
    max_prob = 0.0
    for w, idx in zip(weights, top_indices):
        cwe_code = IDX_TO_CWE[idx]
        expert_prob = float(row[f"{cwe_code}_pred_proba"])
        max_prob = max(max_prob, expert_prob)
        fused_score += w * expert_prob

    return fused_score, max_prob


# -----------------------------
# Evaluation
# -----------------------------
def run_eval(valset: pd.DataFrame,
            testset: pd.DataFrame,
            dynamic_threshold: bool,
            use_topk: bool,
            topk: int,
            topp: float,
            min_expert_num: int):
    """
    Evaluate a routing strategy on val/test sets.

    IMPORTANT: This preserves the original behavior:
      - Predictions are produced by thresholding 'max_prob' (OR-style proxy),
        not by thresholding the fused mixture score.
      - When dynamic_threshold=True, the best threshold is searched using fused_score
        but applied to max_prob (this may be unintended, but we keep it unchanged).
    """
    scorer = (lambda r: compute_topk_softmax_score(r, topk=topk)) if use_topk \
             else (lambda r: compute_topp_softmax_score(r, topp=topp, min_expert_num=min_expert_num))

    if not dynamic_threshold:
        testset["fused_score"], testset["max_prob"] = zip(*testset.apply(scorer, axis=1))
        thr = 0.5
        testset["pred"] = (testset["max_prob"] >= thr).astype(int)
        print(f"[TEST] dynamic_threshold={dynamic_threshold} routing={'topk' if use_topk else 'topp'} "
              f"topk={topk} topp={topp} min_expert_num={min_expert_num} thr={thr}")
        print(classification_report(testset["target"], testset["pred"], digits=3))
        return

    # dynamic threshold mode
    valset["fused_score"], valset["max_prob"] = zip(*valset.apply(scorer, axis=1))

    best_thr, best_f1, best_prec, best_rec, best_acc = calculate_max_f1_threshold(valset, col="fused_score")
    print(f"[VAL] best_thr (searched on fused_score) = {best_thr:.6f}")

    # NOTE: preserve original behavior: apply threshold to max_prob
    valset["pred"] = (valset["max_prob"] >= best_thr).astype(int)
    print(f"[VAL] dynamic_threshold={dynamic_threshold} routing={'topk' if use_topk else 'topp'} "
          f"topk={topk} topp={topp} min_expert_num={min_expert_num}")
    print(classification_report(valset["target"], valset["pred"], digits=3))

    testset["fused_score"], testset["max_prob"] = zip(*testset.apply(scorer, axis=1))
    testset["pred"] = (testset["max_prob"] >= best_thr).astype(int)
    print(f"[TEST] dynamic_threshold={dynamic_threshold} routing={'topk' if use_topk else 'topp'} "
          f"topk={topk} topp={topp} min_expert_num={min_expert_num} thr={best_thr:.6f}")
    print(classification_report(testset["target"], testset["pred"], digits=3))


def parse_args():
    p = argparse.ArgumentParser(description="Sweep Top-k / Top-p routing strategies for MoE inference.")
    p.add_argument("--selected_model", type=str, default="microsoft/codebert-base")
    p.add_argument("--expert_root", type=str, default="../models/output_codebert-base_seed0")
    p.add_argument("--router_root", type=str, default="../models/output_codebert-base_seed42")
    p.add_argument("--router_folder", type=str, default="focal_ep20_bs512_eval_f1_macro_gamma1")

    p.add_argument("--dataset_root", type=str, default="../data")
    p.add_argument("--topp", type=float, default=0.8)
    p.add_argument("--dynamic_threshold", action="store_true")

    return p.parse_args()


def main():
    args = parse_args()

    # Load ground-truth labels
    val_df = pd.read_parquet(f"{args.dataset_root}/val_cwe.parquet")
    test_df = pd.read_parquet(f"{args.dataset_root}/test_cwe.parquet")

    # Load expert probabilities
    binary_val = load_binary_probs(args.expert_root, CWE_LIST, split="val")
    binary_test = load_binary_probs(args.expert_root, CWE_LIST, split="test")

    # Load router probabilities
    router_val = load_router_probs(args.router_root, args.router_folder, split="val")
    router_test = load_router_probs(args.router_root, args.router_folder, split="test")

    # Merge into evaluation tables
    valset = pd.concat([binary_val, router_val, val_df[["level1", "target"]]], axis=1)
    testset = pd.concat([binary_test, router_test, test_df[["level1", "target"]]], axis=1)

    # Sweep fixed Top-k routing (k=1..N)
    for k in range(1, len(CWE_LIST) + 1):
        run_eval(
            valset=valset.copy(),
            testset=testset.copy(),
            dynamic_threshold=args.dynamic_threshold,
            use_topk=True,
            topk=k,
            topp=args.topp,
            min_expert_num=k,  # keep your original pattern: min_expert_num aligned with k
        )

    # Sweep Top-p routing with different minimum expert constraints
    for m in range(1, len(CWE_LIST) + 1):
        run_eval(
            valset=valset.copy(),
            testset=testset.copy(),
            dynamic_threshold=args.dynamic_threshold,
            use_topk=False,
            topk=1,            # unused in top-p mode, kept for signature compatibility
            topp=args.topp,
            min_expert_num=m,
        )


if __name__ == "__main__":
    main()
