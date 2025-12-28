#!/bin/bash

root="./data"
save_path="./results"
log_file="${save_path}/train.log"

mkdir -p $save_path

python train_sew.py \
    --method sew_post \
    --arch vgg16_bn \
    --dataset Cifar10 \
    --target_label 0 \
    --lr 0.1 \
    --batch_size 128 \
    --epoch 100 \
    --workers 4 \
    --gpu 0 \
    --root ${root} \
    --save_path ${save_path} \
    2>&1 | ts >> ${log_file}