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

export LOG_DIR=./outputs/FacilityArea5Dataset/8_classes/$TIME

# Save the experiment detail and dir to the common log file
mkdir -p $LOG_DIR

LOG="$LOG_DIR/$TIME.txt"

CUDA_LAUNCH_BLOCKING=1 python -m main \
    --resume $LOG_DIR/../test \
    --is_train False \
    --train_phase test \
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
<<<<<<< HEAD
    --max_iter 500 \
    --data_aug_color_trans_ratio 0.05 \
    --data_aug_color_jitter_std 0.005 \
    $3 2>&1 | tee -a "$LOG"
#--resume $LOG_DIR/../test \
=======
    --max_iter 200 \
    --data_aug_color_trans_ratio 0.05 \
    --data_aug_color_jitter_std 0.005 \
    --val_freq 1 \
    --train_phase train \
    #--resume $LOG_DIR/.. \
    --is_train True \
    $3 2>&1 | tee -a "$LOG"
>>>>>>> 5c7b8111948ac609489e36edb941e786de79f25f
