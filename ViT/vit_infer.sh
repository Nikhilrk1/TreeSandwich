#!/usr/bin/env bash

# Call 0 epoch train to generate data from APIs
python ViT/vit.py \
  --mode train \
  --csv "ViT/tree_line_distance_compare 2.csv" \
  --images-dir ViT/images \
  --meta-cache-csv "ViT/tree_line_distance_compare 2_meta_cache.csv" \
  --save-path /tmp/dummy_1.pt \
  --epochs 0

python ViT/vit.py \
  --mode infer \
  --csv "ViT/tree_line_distance_compare 2.csv" \
  --images-dir ViT/images \
  --model-path ViT/models \
  --meta-cache-csv "ViT/tree_line_distance_compare_meta_cache.csv" \
  --inference-output-csv "ViT/future_predictions_2.csv"
