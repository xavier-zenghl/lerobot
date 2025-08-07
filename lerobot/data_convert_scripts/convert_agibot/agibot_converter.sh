#!/bin/bash
export PYTHONWARNINGS="ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning"

TASK_IDS=(
    429
)

for TASK_ID in "${TASK_IDS[@]}"; do
    python convert_agibot_to_so3_lerobot.py --src_path /home/xavierzeng/data/agibot --task_id ${TASK_ID} --tgt_path /home/xavierzeng/data/lerobot_datasets/agibot  
done