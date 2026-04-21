---
id: a3s7d1f5g9h2j6k4l8p0
title: Model ResNet18
desc: CNN baseline status, settings, and outputs
updated: 1776687045747
created: 1776687045747
---

# Model ResNet18

## Purpose

Track the CNN baseline used in the dissertation comparison.

## Configuration

- Backbone: `ResNet18`
- Pretrained: `true`
- Source: `torchvision_imagenet1k`
- Optimizer: `AdamW`
- Learning rate: `1e-4`
- Weight decay: `0.01`
- Epochs: `30`
- Batch size: `64`

## Current Results

- Test accuracy: `0.9474`
- Test macro-F1: `0.9457`

## XAI Used

- Grad-CAM
- Integrated Gradients
- SmoothGrad

## Saved Outputs

- Checkpoint: `yc432/checkpoints/resnet18_best.pth`
- Metrics: `yc432/results/metrics/resnet18_test_metrics.json`
- XAI metrics: `yc432/results/metrics/resnet18_xai_quant_metrics.json`

## Note

- ResNet18 is the strongest completed CNN baseline already available in the repo

## Linked Notes

- [[xai.gradcam]]
- [[xai.integrated-gradients]]
- [[xai.smoothgrad]]
- [[results.comparison]]
