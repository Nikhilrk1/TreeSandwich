#!/usr/bin/env bash

python ViT/vit.py \
  --csv ViT/tree_line_distance_compare.csv \
  --images-dir ViT/images \
  --save-path ViT/models \
  --epochs 10 \
  --batch-size 40 \
  --lr 1e-4 \
  --train-ratio .8 \
  --seed 2026

