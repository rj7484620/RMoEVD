# RMoEVD
This document will be updated by 11:59 AM UTC on January 31st.
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

After downloading, run `data/read_bigvul.ipynb`.