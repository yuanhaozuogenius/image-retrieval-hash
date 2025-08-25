#!/bin/bash

# ===== 运行配置 =====
#DATA_PATH="D:/Datasets/cifar10"
DATA_PATH="../dataset/cifar/"
DATA_NAME="cifar10-1"
WORD2VEC_FILE="../data/cifar10/cifar10_bert768_word2vec.pkl"
BATCH_SIZE=64
HASH_BIT=64
EPOCHS=90
RADIUS=1000

# ===== 启动训练 =====
python train.py \
  --data_path $DATA_PATH \
  --data_name $DATA_NAME \
  --word2vec_file $WORD2VEC_FILE \
  --epochs $EPOCHS \
  --fixed_weight \
  --center_update \
  --R $RADIUS \
  --batch_size $BATCH_SIZE \
  --hash_bit $HASH_BIT