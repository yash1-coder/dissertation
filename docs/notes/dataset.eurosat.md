---
id: p6b1n9q4r7s2t5u8v3w0
title: Dataset EuroSAT
desc: Core dataset facts and project-specific usage
updated: 1776687045747
created: 1776687045747
---

# Dataset EuroSAT

## Purpose

Keep the key dataset facts used across the dissertation in one place.

## Dataset Facts

- Dataset: EuroSAT RGB
- Classes: 10
- Image size in config: 64 x 64 RGB
- Data directory: `yc432/data/2750`

## Class Mapping

- `AnnualCrop`: 0
- `Forest`: 1
- `HerbaceousVegetation`: 2
- `Highway`: 3
- `Industrial`: 4
- `Pasture`: 5
- `PermanentCrop`: 6
- `Residential`: 7
- `River`: 8
- `SeaLake`: 9

## Why It Matters

- All three model branches use the same EuroSAT label space
- The notebook builds one metadata table before split generation and training
- XAI comparisons only make sense if the dataset setup stays fixed

## Linked Notes

- [[dataset.splits]]
- [[models.resnet18]]
- [[models.vit-tiny]]
- [[models.vision-mamba]]
