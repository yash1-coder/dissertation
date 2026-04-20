# PROJECT_GUIDE

# What this project does

This dissertation project studies **satellite image classification and explainability** on the EuroSAT RGB dataset. The notebook `CNN.ipynb` is the main executable workflow. It takes the reader from environment setup and dataset verification through exploratory analysis, model training, test evaluation, explainability outputs, and a final cross-model comparison.

The practical aim is twofold:

1. measure how well the implemented models classify EuroSAT land-use categories
2. examine how their predictions can be interpreted using visualization-based explainability methods

# System architecture 

- **Dataset and inputs:** EuroSAT RGB imagery under `data/2750` and the central config file provide the raw inputs and execution settings for the workflow.
- **Split generation and reproducibility:** `CNN.ipynb` validates the dataset and creates reusable train/validation/test split artefacts plus class mapping and split hashes.
- **Main notebook pipeline:** `CNN.ipynb` is the orchestration hub for setup, EDA, model execution, evaluation/XAI, and the final comparison stage.
- **Model branches:** ResNet18, ViT Tiny, and Vision Mamba / Vim are parallel architecture families that reuse the same split files so comparison remains fair and methodologically aligned.
- **Evaluation outputs:** each model writes checkpoints, predictions, metrics, classwise summaries, confusion matrices, and training curves into the shared `checkpoints/`, `results/metrics/`, and `results/figures/` folders.
- **Explainability outputs:** the project produces qualitative and quantitative XAI outputs in `results/heatmaps/` and `results/failure_cases/`, using Grad-CAM for ResNet18, attention rollout for ViT Tiny, and Integrated Gradients/SmoothGrad for Vim.
- **Final comparison:** the notebook aggregates predictive and explainability outputs into `results/metrics/model_comparison_summary.csv`, with an optional dashboard artefact for reader-facing synthesis.
- **Supporting files:** `PROJECT_GUIDE.md`, `configs/master_config.yaml`, and `src/models/vim.py` document, configure, and implement the current workflow. `src/models/vim.py` includes a pure-PyTorch CPU fallback that activates automatically when `mamba-ssm` is unavailable, enabling local architecture testing on any platform; full CUDA training still requires Linux with `requirements-vim.txt` installed.

## Which models are implemented

The notebook now implements three model families:

- **ResNet18**
  CNN baseline for EuroSAT classification, with training, evaluation, confusion analysis, qualitative XAI panels, and quantitative XAI metrics.

- **ViT Tiny (`vit_tiny_patch16_224`)**
  Transformer baseline for the same task and split, with training, evaluation, confusion analysis, qualitative XAI panels, and quantitative XAI metrics.

- **Vision Mamba / Vim-Tiny**
  State-space baseline for the same task and split, with training, evaluation, confusion analysis, qualitative XAI panels, and quantitative XAI metrics.

## What “Vim” means in this repository

In this project, **Vim** refers to the **Vision Mamba** architecture family introduced in the paper:

- *Vision Mamba: Efficient Visual Representation Learning with Bidirectional State Space Model*

and associated with the official repository:

- `hustvl/Vim`

The local implementation in this repository is a **classification-focused Vim-Tiny adaptation** of that official design. It keeps the core Vision Mamba structure that matters for this dissertation:

- image-to-patch embedding
- learnable class token
- absolute positional embedding
- stacked **bidirectional Mamba** sequence mixing blocks
- final normalization and classifier head

It is intentionally **not** `timm`'s `mambaout_*` family, because that is a different architecture family and would not be a clean Vision Mamba baseline for this dissertation comparison.

## Exact implementation choice

The local file implementing Vim is:

- `src/models/vim.py`

This file is adapted from the official Vim architecture path but uses the upstream `mamba-ssm` runtime already referenced in the implementation plan.

Important technical note:

- the public `hustvl/Vim` code path expects a custom bi-Mamba style interface that is not exposed by the plain upstream `mamba-ssm` package
- this repository therefore implements **bidirectional sequence mixing explicitly** by combining a forward Mamba pass and a reversed-sequence Mamba pass inside each Vim block
- this preserves the dissertation-critical architectural idea of **Vision Mamba as a bidirectional state-space image model**, while remaining practical and maintainable inside the current project

This is the strongest workable Vision Mamba implementation path for this repository without introducing an additional custom fork of the Mamba runtime.

## Why Vim is compatible with the current pipeline

Vim is integrated as a **third additive pipeline**, not a destructive rewrite.

It is compatible with the current dissertation workflow because it:

- reuses the existing `train.csv`, `val.csv`, and `test.csv` split files
- reuses the same EuroSAT class mapping file
- follows the same train/validation/test evaluation discipline as ResNet18 and ViT Tiny
- writes outputs into the same `checkpoints/`, `results/metrics/`, `results/figures/`, `results/heatmaps/`, and `results/failure_cases/` structure
- exposes a standard image-classification interface: input tensor in, class logits out
- supports architecture-agnostic XAI methods already used elsewhere in the dissertation

## Runtime and dependency requirements

The base repository dependencies remain unchanged for ResNet18 and ViT Tiny. Vision Mamba uses a separate dependency file:

- `requirements-vim.txt`

Install command (Linux/CUDA server only):

- `pip install --no-build-isolation -r requirements-vim.txt`

Additional dependencies required for full Vim training:

- `causal-conv1d>=1.4.0`
- `mamba-ssm`

Why they are required:

- `mamba-ssm` provides the underlying state-space sequence mixer used inside the Vision Mamba blocks
- `causal-conv1d` is a runtime dependency commonly required by Mamba builds

### Two execution modes

**CPU fallback (local / macOS / no CUDA)**

When `mamba-ssm` cannot be imported (e.g. on macOS or any machine without a CUDA build environment), `src/models/vim.py` automatically activates `_MambaCPUFallback` — a pure-PyTorch depthwise Conv1d + SiLU gated approximation with the same `(B, L, D) → (B, L, D)` interface. This lets the full model be instantiated, forward-passed, and inspected locally without any additional installation steps. It is **not** equivalent to the full selective-scan kernel and should not be used for final dissertation results; it exists solely for local architecture smoke-testing and notebook development.

`vim_runtime_status()` returns `(True, "Vim running with CPU fallback ...")` in this mode.

**Full CUDA training (Linux remote server)**

The dissertation Vim pipeline is intended for:

- **Linux**
- **NVIDIA CUDA**

On the remote server, install `requirements-vim.txt` as above. `mamba_ssm` will be imported directly and `_MambaCPUFallback` will not be used. `vim_runtime_status()` returns `(True, "Vim runtime looks available.")`.

The notebook includes guards so unsupported environments do not overwrite the existing ResNet18 or ViT Tiny artefacts.

## What outputs are generated

### 1. Dataset and split artefacts

Located mainly under `data/`:

- `data/splits/train.csv`
- `data/splits/val.csv`
- `data/splits/test.csv`
- `data/splits/class_mapping.json`
- `data/splits/split_hashes.json`

These files define the exact train/validation/test partition used by all three model families and support reproducibility checks.

### 2. EDA artefacts

Located under `data/eda/`:

- `class_distribution.png`
- `sample_images_grid.png`
- `channel_histograms.png`
- `split_stratification_check.png`
- `channel_statistics.json`

### 3. Checkpoints

Located under `checkpoints/`:

- `resnet18_best_10ep.pth`
- `resnet18_best.pth`
- `vit_tiny_best.pth`
- `vim_tiny_best.pth`

### 4. Model evaluation outputs

Located under `results/metrics/` and `results/figures/`:

#### ResNet18

- `resnet18_test_metrics.json`
- `resnet18_test_predictions.csv`
- `resnet18_classwise_metrics.csv`
- `resnet18_xai_quant_metrics.json`
- `training_curves.png`
- `resnet18_confusion_matrix_counts_norm.png`

#### ViT Tiny

- `vit_tiny_test_metrics.json`
- `vit_tiny_test_predictions.csv`
- `vit_tiny_classwise_metrics.csv`
- `vit_tiny_train_history.json`
- `vit_tiny_xai_quant_metrics.json`
- `vit_tiny_training_curves.png`
- `vit_tiny_confusion_matrix_counts_norm.png`

#### Vision Mamba / Vim-Tiny

- `vim_test_metrics.json`
- `vim_test_predictions.csv`
- `vim_classwise_metrics.csv`
- `vim_train_history.json`
- `vim_xai_quant_metrics.json`
- `vim_training_curves.png`
- `vim_confusion_matrix_counts_norm.png`

### 5. Explainability figures

Located under `results/heatmaps/` and `results/failure_cases/`:

#### ResNet18

- `xai_trueclass_*.png`
- `xai_sample_*.png`
- `misclassified_xai_*.png`

#### ViT Tiny

- `vit_xai_sample_*.png`
- `vit_misclassified_xai_*.png`

#### Vision Mamba / Vim-Tiny

- `vim_xai_sample_*.png`
- `vim_misclassified_xai_*.png`

### 6. Final comparison summary

Located under `results/metrics/`:

- `model_comparison_summary.csv`

This table now includes:

- ResNet18
- ViT Tiny
- Vision Mamba
- delta rows comparing Vim against the other model families

## How to interpret the main figures

### Training curves

Files:

- `results/figures/training_curves.png`
- `results/figures/vit_tiny_training_curves.png`
- `results/figures/vim_training_curves.png`

These plots show loss, accuracy, and macro-F1 across epochs.

How to read them:

- falling training and validation loss usually indicates that optimization is progressing
- rising validation accuracy and macro-F1 suggest improving generalization
- a growing gap between training and validation performance can indicate overfitting

### Confusion matrices

Files:

- `results/figures/resnet18_confusion_matrix_counts_norm.png`
- `results/figures/vit_tiny_confusion_matrix_counts_norm.png`
- `results/figures/vim_confusion_matrix_counts_norm.png`

These figures show where predictions are correct and where systematic class confusions remain.

### Qualitative XAI figures

These figures should be treated as **interpretive aids**, not causal proof.

#### ResNet18 panels

Typical panels include:

- original image
- Grad-CAM
- Integrated Gradients
- SmoothGrad

#### ViT Tiny panels

Typical panels include:

- original image
- Integrated Gradients
- Attention Rollout
- overlay visualization

#### Vision Mamba panels

Typical panels include:

- original image
- Integrated Gradients
- SmoothGrad
- overlay visualization

Why these methods are used for Vim:

- **Integrated Gradients** is architecture-agnostic and provides a consistent cross-model attribution baseline
- **SmoothGrad** is also architecture-agnostic and helps reduce gradient noise for state-space models
- **Grad-CAM** is not used because Vim is not a CNN with a late convolutional feature-map target layer in the same sense as ResNet18
- **Attention Rollout** is not used because Vim is not an attention-based transformer model

### Quantitative XAI metrics

Files:

- `resnet18_xai_quant_metrics.json`
- `vit_tiny_xai_quant_metrics.json`
- `vim_xai_quant_metrics.json`

The notebook reports three main metrics for all implemented models:

- **Faithfulness mean drop**
- **Stability mean cosine**
- **Sensitivity mean top minus random**

The quantitative Vim evaluation reuses the same **Integrated Gradients-based** protocol used for ResNet18 and ViT Tiny so the three-way comparison remains methodologically aligned.

## Suggested reading order

1. Start with `CNN.ipynb` as the main executable dissertation workflow.
2. Review dataset setup and EDA before interpreting any model output.
3. Read the ResNet18 section.
4. Read the ViT Tiny section.
5. Read the Vision Mamba / Vim-Tiny section.
6. Finish with `model_comparison_summary.csv`.

## Scope and limitation note

This repository now contains a real Vision Mamba implementation path, but one important limitation should be kept in mind:

- the local Vim module is **adapted from the official Vision Mamba design**, not a byte-for-byte copy of the official repository runtime stack
- this is because the official code path relies on a custom bi-Mamba interface that is not directly available from the plain upstream `mamba-ssm` package
- the dissertation implementation keeps the essential Vision Mamba design choice by using **bidirectional Mamba mixing** explicitly inside each block

For this project, that tradeoff is appropriate because it keeps the model academically faithful, technically transparent, and compatible with the existing EuroSAT comparison pipeline.
