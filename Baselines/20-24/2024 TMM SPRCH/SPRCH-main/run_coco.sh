#!/bin/bash

# ===== 启动训练 =====
python main.py \
  --data_path "D:/Datasets/coco2017" \
  --data_name "coco" \
  --data_class 80 \
  --epochs 100 \
  --batchSize 64 \
  --binary_bits 64 \
  --lr 1e-5
