# RMoEVD
## Dataset Preparation

To ensure a strict and fair comparison with MoEVD, we use **exactly the same BigVul dataset release and preprocessing pipeline provided by the original MoEVD artifact**.

The processed dataset files used in our experiments are **not stored in this repository** due to size and licensing constraints. Instead, they must be downloaded from the official MoEVD artifact release:

**MoEVD Artifact (Zenodo):**  
https://zenodo.org/records/11661787

After downloading the dataset package, place the following files under `data/`

| File | Description |
|------|-------------|
| `train.parquet` | BigVul train split |
| `val.parquet`   | BigVul validation split |
| `test.parquet`  | BigVul test split |
| `read_bigvul.ipynb` | MoEVD data process script |
| `expert_train.py` | MoEVD expert train script |
| `router_train.py` | MoEVD router train script |
| `inference.py` | MoEVD inference script |

After downloading, run `data/read_bigvul.ipynb`.

## Reproducing RQ1 and RQ2 Results

This section describes how to reproduce the experiments related to **RQ1 (Expert Behavior Analysis)** and **RQ2 (Impact of Router Misassignment)** in MoEVD framework.

To obtain the RQ1 and RQ2 results, you must first reproduce the **expert training** and **router training** stages of MoEVD. After the models are trained, the statistical analysis scripts under `src/stats/` can be used to generate the final results.

### 1. Expert Training (MoE Experts)

Each expert is trained to specialize in a specific CWE type.

Run:

```bash
python expert_train.py
```

### Parameters

| Parameter    | Description                                                                                                           |
| ------------ | --------------------------------------------------------------------------------------------------------------------- |
| `cwe`        | The CWE category that the expert is trained to detect (positive class).                                               |
| `force_cwe`  | If enabled, all other CWE types are treated as negative samples. This enforces stricter specialization of the expert. |
| `model_name` | Backbone model name from HuggingFace (non-quantized models supported).                                                |


### 2. Router Training

The router determines which expert(s) should process each input sample.

Run:

```bash
python router_train.py
```

### Parameters

| Parameter    | Description                                                                                                                |
| ------------ | -------------------------------------------------------------------------------------------------------------------------- |
| `model_name` | Backbone model name from HuggingFace (non-quantized models supported).                                                     |
| `epochs`     | Number of training epochs. Should be adjusted according to dataset size.                                                   |
| `ros`        | If enabled, Random Over Sampling (ROS) is used across CWE types instead of Focal Loss. Both strategies perform comparably. |


### 3. Generating RQ1 and RQ2 Results

After expert models and the router model are trained, use the analysis scripts located in:

```
src/stats/
```

#### Scripts

| Script                    | Purpose                                                                            |
| ------------------------- | ---------------------------------------------------------------------------------- |
| `RQ1-experts_analysis.py` | Evaluates expert specialization and effectiveness (RQ1).                           |
| `RQ2-router_analysis.py`  | Analyzes router decisions, routing distribution, and misassignment behavior (RQ2). |
| `heatmap.py`              | Generates visualization heatmaps for experts.                |
| `statistic_result.md`     | Output summary of computed statistics.                                             |


## Relaxed (Abstract) Expert Construction (RQ3)

This part implements the improved expert design in the paper, where expert specialization is no longer strictly bound to predefined CWE labels.

 `generate_router_based_targets.py`

**Method correspondence:**  
Implements the data reconstruction stage for *abstract expert formulation*.

**Function:**  
Uses router/expert predictions to generate `new_target` labels for all samples. These labels reflect routing behavior and are used to rebuild supervision signals for improved experts.

 `train_reconstructed_expert.py`

**Method correspondence:**  
Implements *relaxed expert construction* (abstract experts) described in the improved MoE design.

**Function:**  
Trains new binary experts using reconstructed supervision:

- **Single mode:** refined expert guided by router predictions  
- **Merged mode:** abstract-class expert where multiple CWEs form one latent vulnerability group  

Handles relabeling, model training, evaluation, and probability output.

---

**Usage Order:**

1. Run `generate_router_based_targets.py` to reconstruct dataset labels from router behavior.  
2. Run `train_reconstructed_expert.py` to train reconstructed or abstract experts.

## Probabilistic Routing Strategy (RQ3)
`sweep_topk_topp_routing`

This script evaluates MoE inference under different expert routing strategies.

It combines router probabilities (expert selection) and expert binary outputs (vulnerability confidence) to produce final predictions.

### Method Correspondence

Implements the routing-side design:

- Top-k routing  
- Top-p routing  
- Softmax-weighted expert aggregation  
- OR-style decision rule  

### Function

For each routing setup, it selects experts, aggregates their outputs, applies a binary decision rule, and reports evaluation metrics.

Used to study how routing breadth affects detection performance.

