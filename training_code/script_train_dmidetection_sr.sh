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
CKPT_DIR="/workspace/training_code/checkpoints"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_CKPT_DIR="${CKPT_DIR}/${EXP_NAME}"

# Optional: if you have an old checkpoint directory, set this and it will be copied first.
# Example: SOURCE_CKPT_DIR="/mnt/host_ckpts/coco_ae_aligned_sr"
SOURCE_CKPT_DIR=""

mkdir -p "${TARGET_CKPT_DIR}"

if [ -n "${SOURCE_CKPT_DIR}" ] && [ -d "${SOURCE_CKPT_DIR}" ]; then
  cp -Rn "${SOURCE_CKPT_DIR}/." "${TARGET_CKPT_DIR}/"
fi

# Auto-resume from latest numeric epoch in TARGET_CKPT_DIR (model_epoch_XX.pth).
RESUME_EPOCH="$(ls "${TARGET_CKPT_DIR}"/model_epoch_[0-9]*.pth 2>/dev/null \
  | sed -E 's|.*/model_epoch_([0-9]+)\.pth|\1|' \
  | sort -n \
  | tail -1)"

CMD=(
  python "${SCRIPT_DIR}/train_dmidetection_sr.py"
  --name "${EXP_NAME}"
  --sr_dataroot "${SR_DATASET_ROOT}"
  --checkpoints_dir "${CKPT_DIR}"
  --arch res50nodown
  --cropSize 96
  --norm_type resnet
  --resize_size 256
  --resize_ratio 0.75
  --blur_sig 0.0,3.0
  --cmp_method cv2,pil
  --cmp_qual 30,100
  --resize_prob 0.2
  --jitter_prob 0.8
  --colordist_prob 0.2
  --cutout_prob 0.2
  --noise_prob 0.2
  --blur_prob 0.5
  --cmp_prob 0.5
  --rot90_prob 1.0
  --batch_size 32
  --earlystop_epoch 5
  --seed 14
  --stay_positive clamp
  --fix_backbone
  --final_dropout 0.0
  --batched_syncing
  --use_inversions
)

if [ -n "${RESUME_EPOCH}" ]; then
  echo "[Resume] model_epoch_${RESUME_EPOCH}.pth"
  CMD+=(--continue_epoch "${RESUME_EPOCH}")
fi

CUDA_VISIBLE_DEVICES=0 "${CMD[@]}"
