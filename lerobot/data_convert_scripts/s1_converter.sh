#!/bin/bash

BASE_DIR="/home/xavierzeng/workspace/data/"
ROOT_PATH="/home/xavierzeng/workspace/data/test/"
SCRIPT_DIR="/home/xavierzeng/workspace/code/lerobot/lerobot/data_convert_scripts"
GPU_IDX=0

SUBDIR_LIST=(
  "0731_desktop_Storage_S8"
)
REPO_ID_LIST=(
  "lerobot_so3_data_30hz"
)
TASK_DESCRIPTION_LIST=(
  "A robot is positioned in front of the checkout counter, where three different types of items and a shopping bag are placed. Packing in the supermarket."
)

cd "$SCRIPT_DIR" || { echo "❌ Failed to cd into $SCRIPT_DIR"; exit 1; }

for idx in "${!SUBDIR_LIST[@]}"; do
  SUBDIR="${SUBDIR_LIST[$idx]}"
  RAW_PATH="${BASE_DIR}${SUBDIR}"
  ROOT_PATH="${ROOT_PATH}${SUBDIR}"
  REPO_ID="${REPO_ID_LIST[$idx]}"
  HDF5_DIR="${RAW_PATH}/hdf5"

  echo "========== 处理 $SUBDIR =========="
  
  # TODO----------------- 新增步骤0: 检测目标hdf5是否包含prompt键值 ------------------
  # echo "🔵 [Step0] 检测 $HDF5_DIR 下的 HDF5 文件是否包含 prompt 键值"
  # python check_hdf5.py "$HDF5_DIR"
  # if [[ $? -ne 0 ]]; then
  #     echo "❌ check_hdf5.py 检测失败，退出"; exit 1
  # fi

  # ------------------ 步骤2: 生成数据集 ------------------
  CUDA_VISIBLE_DEVICES=$GPU_IDX \
    python convert_hdf5_to_lerobot.py \
      --description "${TASK_DESCRIPTION_LIST[$idx]}" \
      --raw-path "$RAW_PATH" \
      --root-path "$ROOT_PATH" \
      --dataset-repo-id "$REPO_ID" \
      --hdf5_compressed True \
      --split False

  echo "✅ 处理完成: ${SUBDIR}"
done
echo "🎉🎉🎉 所有文件夹全部处理完成！"