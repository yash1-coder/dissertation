---
id: k4r8m2c7d1f9g5h3j6l0
title: Project Environment
desc: Runtime requirements and blockers for local execution
updated: 1776687045747
created: 1776687045747
---

# Project Environment

## Purpose

Capture what runs locally now and what is blocked by runtime constraints.

## Working Paths

- ResNet18 runs in the base project environment
- ViT Tiny runs in the base project environment
- Shared config is in `yc432/configs/master_config.yaml`

## Current Blocker

- Vision Mamba depends on `mamba-ssm` and `causal-conv1d`
- Vim path is intended for Linux + NVIDIA CUDA
- Current repo has Vim code, but no saved Vim checkpoint or metrics artefacts yet

## Relevant Files

- `yc432/requirements.txt`
- `yc432/requirements-vim.txt`
- `yc432/src/models/vim.py`

## Practical Next Step

- Use a Linux/CUDA runtime
- Install Vim extras with `pip install --no-build-isolation -r yc432/requirements-vim.txt`
- Run the Vim notebook section without overwriting existing ResNet18 or ViT outputs

## Linked Notes

- [[models.vision-mamba]]
- [[tasks.next]]
