---
id: r3t7y1u5i9o2p6a0s4d8
title: Results Comparison
desc: Cross-model summary and current gaps
updated: 1776687045747
created: 1776687045747
---

# Results Comparison

## Purpose

Keep the current model-vs-model summary close to the dissertation notes.

## Current Saved Comparison

- Source: `yc432/results/metrics/model_comparison_summary.csv`
- Saved rows currently cover ResNet18 and ViT Tiny
- Vision Mamba comparison is still pending execution

## Current Headline Numbers

- ResNet18: accuracy `0.9474`, macro-F1 `0.9457`
- ViT Tiny: accuracy `0.9862`, macro-F1 `0.9856`
- ViT minus ResNet18: accuracy `+0.0388`, macro-F1 `+0.0399`

## XAI Summary

- ViT Tiny has better saved stability and sensitivity scores than ResNet18
- ResNet18 has a higher saved faithfulness mean drop than ViT Tiny
- The final discussion should explain what each metric means before claiming one model is more explainable

## Gap To Close

- Add the Vim row after Linux/CUDA execution
- Update comparison text, tables, and figures once Vim metrics exist

## Linked Notes

- [[models.resnet18]]
- [[models.vit-tiny]]
- [[models.vision-mamba]]
- [[tasks.next]]
