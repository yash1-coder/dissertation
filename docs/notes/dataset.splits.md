---
id: z1x5c9v3b7n2m6q4w8e0
title: Dataset Splits
desc: Reproducible train, validation, and test split record
updated: 1776687045747
created: 1776687045747
---

# Dataset Splits

## Purpose

Record the exact reusable split setup for fair model comparison.

## Split Setup

- Train / val / test ratio: 0.70 / 0.15 / 0.15
- Split seed: 42
- Stratified sampling is used in `yc432/CNN.ipynb`

## Current Counts

- Train: 18,900
- Validation: 4,050
- Test: 4,050

## Saved Artefacts

- `yc432/data/splits/train.csv`
- `yc432/data/splits/val.csv`
- `yc432/data/splits/test.csv`
- `yc432/data/splits/class_mapping.json`
- `yc432/data/splits/split_hashes.json`

## Current Hashes

- Train: `d45f05217b5fbd2b`
- Val: `11e0935a15995137`
- Test: `f6d60a70c3b8b5ac`

## Practical Reminder

- Do not regenerate splits mid-writeup unless the change is intentional and documented

## Linked Notes

- [[dataset.eurosat]]
- [[results.comparison]]
