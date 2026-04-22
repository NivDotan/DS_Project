# Multi-Class Solar Flare Prediction

This repository contains a data science project for predicting solar flare severity from multivariate time-series measurements of solar active regions.

The project focuses on a difficult forecasting setting: strong class imbalance, noisy physical measurements, overlapping flare classes, and strict time-order constraints. The final notebook presents the full project story, including preprocessing, temporal windowing, model comparisons, staged classifiers, LightGBM baselines, temporal models, and interpretability/overlap analysis.

## Main Notebook

Open this notebook first:

```text
Multi-Class Solar Flare Prediction Notebook.ipynb
```

The notebook is written as the main project report. It includes the motivation, dataset explanation, modeling workflow, experiment summaries, figures, and conclusions.

## Project Goal

The goal is to predict the strongest solar flare class expected in a future forecasting horizon using historical measurements from a solar active region.

Classes are treated as a 5-class prediction problem:

```text
quiet, B, C, M, X
```

## Workflow

The project follows this structure:

1. Exploratory data analysis
2. Streaming preprocessing for large files
3. Temporal window generation
4. Feature engineering
5. Class imbalance handling
6. LightGBM baseline experiments
7. Stage-based temporal modeling
8. Evaluation with confusion matrices and per-class metrics
9. Overlap and interpretability analysis

## Models and Methods

The project explores several approaches:

- LightGBM tabular baselines
- BiLSTM temporal models
- Transformer-style temporal gates
- MiniRocket-based temporal features
- Physics-informed and staged classification ideas
- Threshold tuning and post-processing

Not every experimental run is included in full detail. The notebook focuses on the models and configurations that produced meaningful changes, useful failures, or important project insights.

## Included Files

Important files and folders:

```text
Multi-Class Solar Flare Prediction Notebook.ipynb
project_paths.py
section6_runtime_bundle_minimal.py
stage1_w72_notebook_block.py
stage1_multirun_render.py
stage2_notebook_results.py
stage2_section_helpers.py
stage3_notebook_results.py
stage3_matrix_helpers.py
professor_notebook_helpers/
lightgbm_run_inventory_20260412/
analysis_artifacts/
results/
```

The `professor_notebook_helpers/` folder contains source blocks used by the notebook so the final report can be understood without relying on hidden local notebook history.

The `lightgbm_run_inventory_20260412/`, `results/`, and small files under `analysis_artifacts/` contain compact result summaries, selected run metadata, confusion matrices, classification reports, and feature-importance outputs.

## What Is Not Included

Large files are intentionally excluded from the repository:

- Raw data
- Preprocessed partitions
- Model checkpoints
- Large `.parquet`, `.npz`, and cache files
- Full training run folders
- Temporary notebooks and archived experiments

This keeps the repository readable and small enough for GitHub.

## Best LightGBM Baseline

The strongest selected LightGBM baseline in the included analysis is:

```text
Run: W72_H24
Window: 72 hours
Forecast horizon: 24 hours
Experiment: weighted_thresholded
5-class test Macro-F1: 0.364946
```

See:

```text
analysis_artifacts/run_selection/selected_best_lightgbm_run_notes.md
analysis_artifacts/lightgbm_best_run/classification_report_test.csv
analysis_artifacts/lightgbm_best_run/confusion_matrix_test.csv
```

## How to Run

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Then open the notebook:

```bash
jupyter notebook "Multi-Class Solar Flare Prediction Notebook.ipynb"
```

Some cells expect the original raw data or full training-output folders, which are not included because of size. The notebook should be read primarily as a project report with selected compact artifacts included for reproducibility and inspection.

## Data Note

The project is based on solar active-region time-series data from the SWAN-SF benchmark style of forecasting problem. The raw dataset is not stored in this repository. To fully rerun preprocessing and training, place the raw data and generated partitions in the paths expected by the notebook or adapt `project_paths.py` to your local environment.

## Project Outcome

The final result did not fully reach the performance level originally hoped for, especially on the rare and overlapping flare classes. That became an important part of the project: the work shows not only which models perform better, but also where the dataset and class formulation create real limits for prediction.

