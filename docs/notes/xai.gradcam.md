---
id: h4j8k2l6p0q3w7e1r5t9
title: XAI GradCAM
desc: Grad-CAM usage for the CNN baseline
updated: 1776687045747
created: 1776687045747
---

# XAI GradCAM

## Purpose

Record how Grad-CAM is used in this project and where it fits.

## Current Use

- Applied to ResNet18
- Config target layer: `layer4`
- Upsampled to `64 x 64`

## Interpretation

- Treat Grad-CAM as coarse spatial emphasis
- Useful for reader-facing localisation
- Not a pixel-precise explanation

## Scope Note

- Keep Grad-CAM as a CNN-specific method
- Do not force it onto ViT Tiny or Vision Mamba for cross-model consistency

## Linked Notes

- [[models.resnet18]]
- [[xai.integrated-gradients]]
