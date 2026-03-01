#!/usr/bin/env bash

# python ViT/vit.py \
#   --csv ViT/tree_line_distance_compare.csv \
#   --images-dir ViT/images \
#   --save-path ViT/models \
#   --epochs 10 \
#   --batch-size 40 \
#   --lr 1e-4 \
#   --train-ratio .8 \
#   --seed 2026

python ViT/vit.py \
  --mode infer \
  --csv ViT/tree_line_distance_compare.csv \
  --images-dir ViT/images \
  --meta-cache-csv ViT/tree_line_distance_compare_meta_cache.csv \
  --model-path ViT/models \
  --inference-output-csv ViT/future_predictions.csv
