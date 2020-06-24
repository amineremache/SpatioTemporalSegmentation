#!/bin/bash

set -x
# Exit script when a command returns nonzero state
set -e
set -o pipefail

export PYTHONUNBUFFERED="True"
export CUDA_VISIBLE_DEVICES=0

export BATCH_SIZE=${BATCH_SIZE:-1}

#export TIME=$(date +"%Y-%m-%d_%H")
export TIME=$(date +"%Y-%m-%d_%H-%M")
#export TIME=$(date +"%Y-%m-%d_%H-%M-%S")

export LOG_DIR=./outputs/FacilityArea5Dataset/9_classes/$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

CUDA_LAUNCH_BLOCKING=1 python -m main \
    --is_train True \
    --train_phase train \
    --val_freq 1 \
    --test_stat_freq 50 \
    --save_freq 50 \
    --dataset FacilityArea5Dataset \
    --batch_size $BATCH_SIZE \
    --scheduler PolyLR \
    --model Res16UNet14 \
    --conv1_kernel_size 5 \
    --log_dir $LOG_DIR \
    --lr 1e-1 \
    --max_iter 500 \
    --data_aug_color_trans_ratio 0.05 \
    --data_aug_color_jitter_std 0.005 \
    $3 2>&1 | tee -a "$LOG"
#--resume $LOG_DIR/../test \
