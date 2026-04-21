---
id: o2i6u0y4t8r1e5w9q3p7
title: Model ViT Tiny
desc: Transformer baseline status, settings, and outputs
updated: 1776687045747
created: 1776687045747
---

# Model ViT Tiny

## Purpose

Track the transformer baseline used for the EuroSAT comparison.

## Configuration

- Model: `vit_tiny_patch16_224`
- Pretrained: `true`
- Source: `timm_imagenet21k`
- Patch size: `16`
- Optimizer: `AdamW`
- Learning rate: `1e-4`
- Weight decay: `0.01`
- Epochs: `30`
- Batch size: `64`

## Current Results

- Test accuracy: `0.9862`
- Test macro-F1: `0.9856`

## XAI Used

- Integrated Gradients
- Attention rollout for qualitative visualisation
- Same quantitative IG-based protocol as the CNN branch

## Saved Outputs

- Checkpoint: `yc432/checkpoints/vit_tiny_best.pth`
- Metrics: `yc432/results/metrics/vit_tiny_test_metrics.json`
- XAI metrics: `yc432/results/metrics/vit_tiny_xai_quant_metrics.json`

## Note

- ViT Tiny is currently the best completed model in the saved comparison outputs

## Linked Notes

- [[xai.integrated-gradients]]
- [[results.comparison]]
- [[tasks.next]]
