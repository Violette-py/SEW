#!/bin/bash

root="./data"
save_path="./results"
load_path="${save_path}/best.pth"
log_file="${save_path}/spec.log"

python spec_calculator.py \
    --method sew_post \
    --arch vgg16_bn \
    --dataset Cifar10 \
    --target_label 0 \
    --lr 0.01 \
    --batch_size 4 \
    --epoch 400 \
    --workers 4 \
    --gpu 0 \
    --root ${root} \
    --load_path ${load_path} \
    2>&1 | ts >> ${log_file}