#!/usr/bin/env bash

# SR-only detector training (no Rectified pipeline)
# Prerequisite:
# - SR dataset root must have:
#   <SR_DATASET_ROOT>/train/.../0_real,1_fake
#   <SR_DATASET_ROOT>/valid/.../0_real,1_fake
#
# If you need to build SR cache first, use:
# CUDA_VISIBLE_DEVICES=0 python training_code/precompute_sr_cache.py \
#   --layout split_class \
#   --dataset_root /path/to/original_dataset_root \
#   --output_root "${SR_DATASET_ROOT}" \
#   --splits train valid \
#   --classes 0_real 1_fake

SR_DATASET_ROOT="/workspace/data/dmidetection_sr"
EXP_NAME="coco_ae_aligned_sr"
CKPT_DIR="/Users/hwangsolhee/Desktop/mlpr/AlignedForensics/training_code/checkpoints"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

CUDA_VISIBLE_DEVICES=0 python "${SCRIPT_DIR}/train_dmidetection_sr.py" \
  --name "${EXP_NAME}" \
  --sr_dataroot "${SR_DATASET_ROOT}" \
  --checkpoints_dir "${CKPT_DIR}" \
  --arch res50nodown \
  --cropSize 96 \
  --norm_type resnet \
  --resize_size 256 \
  --resize_ratio 0.75 \
  --blur_sig 0.0,3.0 \
  --cmp_method cv2,pil \
  --cmp_qual 30,100 \
  --resize_prob 0.2 \
  --jitter_prob 0.8 \
  --colordist_prob 0.2 \
  --cutout_prob 0.2 \
  --noise_prob 0.2 \
  --blur_prob 0.5 \
  --cmp_prob 0.5 \
  --rot90_prob 1.0 \
  --batch_size 512 \
  --earlystop_epoch 5 \
  --seed 14 \
  --stay_positive clamp \
  --fix_backbone \
  --final_dropout 0.0 \
  --batched_syncing \
  --use_inversions
