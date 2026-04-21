---
id: f2g6h0j4k8l1z5x9c3v7
title: XAI SmoothGrad
desc: Noise-averaged gradient visualisation note
updated: 1776687045747
created: 1776687045747
---

# XAI SmoothGrad

## Purpose

Track the noise-averaged gradient method used to make attribution maps easier to inspect.

## Configuration

- Samples: `50`
- Noise sigma fraction: `0.1`

## Current Use

- Used with the ResNet18 explanation panels
- Planned for Vision Mamba because it stays architecture-agnostic

## Interpretation

- Reduces visual speckle in gradient-based maps
- Best used as a qualitative companion, not as the only evidence in the writeup

## Linked Notes

- [[models.resnet18]]
- [[models.vision-mamba]]
- [[xai.integrated-gradients]]
