#!/bin/bash

python main.py \
    --ds coco \
    --nbit 64 \
    --bs 64 \
    --epochs 100 \
    --codebook-method B \
    --margin 0.3 \
    --seed 59495
