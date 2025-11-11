#!/bin/bash

python train.py \
  --data_path "D:/Datasets/coco2017" \
  --data_name "coco" \
  --word2vec_file "../data/coco/coco_bert768_word2vec.pkl" \
  --epochs 90 \
  --fixed_weight \
  --center_update \
  --R 5000 \
  --batch_size 64 \
  --hash_bit 64 \
  --start_test_epoch 30
