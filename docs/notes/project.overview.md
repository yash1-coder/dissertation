---
id: j2w7n4p8q1s6t3u5v9x0
title: Project Overview
desc: Dissertation scope, workflow, and current repo status
updated: 1776687045747
created: 1776687045747
---

# Project Overview

## Purpose

Track the dissertation as a single notebook-driven workflow for EuroSAT classification and explainability.

## Scope

- Dataset: EuroSAT RGB
- Main workflow: `yc432/CNN.ipynb`
- Architectures: ResNet18, ViT Tiny, Vision Mamba / Vim-Tiny
- Goal: compare classification performance and explanation behaviour across model families

## Current State

- ResNet18 pipeline runs and has saved checkpoints and metrics
- ViT Tiny pipeline runs and has saved checkpoints and metrics
- Vision Mamba is integrated in code but still pending full Linux/CUDA execution

## Key Files

- `yc432/CNN.ipynb`
- `yc432/configs/master_config.yaml`
- `yc432/src/models/vim.py`
- `yc432/results/metrics/model_comparison_summary.csv`

## Linked Notes

- [[project.environment]]
- [[dataset.eurosat]]
- [[dataset.splits]]
- [[models.resnet18]]
- [[models.vit-tiny]]
- [[models.vision-mamba]]
- [[results.comparison]]
- [[tasks.next]]
