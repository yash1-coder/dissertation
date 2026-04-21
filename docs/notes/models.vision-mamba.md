---
id: n5m9b3v7c1x4z8l2k6j0
title: Model Vision Mamba
desc: Vim-Tiny implementation status and execution blocker
updated: 1776687045747
created: 1776687045747
---

# Model Vision Mamba

## Purpose

Track the Vim-Tiny branch separately from the completed ResNet18 and ViT runs.

## Implementation Status

- Local model file exists: `yc432/src/models/vim.py`
- Model name in config: `vim_tiny_patch16_224`
- Pretrained: `false`
- Uses explicit bidirectional Mamba mixing in the local implementation

## Runtime Constraint

- Requires `mamba-ssm`
- Requires `causal-conv1d`
- Intended for Linux + NVIDIA CUDA execution

## Current Repo State

- Vim is integrated into the notebook and config
- No saved `vim_tiny_best.pth` checkpoint is present
- No saved Vim metrics are present in `yc432/results/metrics`

## Why This Matters

- The dissertation comparison is architecturally stronger with a real Vision Mamba branch
- The written discussion should clearly separate code integration from completed experimental execution

## Immediate Next Step

- Run the Vim training and evaluation section in a Linux/CUDA environment
- Save checkpoint, test metrics, XAI metrics, and comparison row

## Linked Notes

- [[project.environment]]
- [[xai.integrated-gradients]]
- [[xai.smoothgrad]]
- [[tasks.next]]
