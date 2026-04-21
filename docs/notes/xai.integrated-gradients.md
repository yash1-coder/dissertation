---
id: y7u1i5o9p3a6s0d4f8g2
title: XAI Integrated Gradients
desc: Cross-model attribution baseline for the dissertation
updated: 1776687045747
created: 1776687045747
---

# XAI Integrated Gradients

## Purpose

Use one architecture-agnostic attribution method across the model families.

## Configuration

- Steps: `50`
- Baseline: `zero`

## Project Role

- Used for ResNet18
- Used for ViT Tiny
- Planned as the main comparable XAI method for Vision Mamba

## Quantitative Role

- Current comparison uses IG-based metrics for:
- faithfulness mean drop
- stability mean cosine
- sensitivity top-minus-random

## Practical Note

- This is the cleanest common XAI baseline across CNN, transformer, and state-space models in this repo

## Linked Notes

- [[models.resnet18]]
- [[models.vit-tiny]]
- [[models.vision-mamba]]
- [[results.comparison]]
