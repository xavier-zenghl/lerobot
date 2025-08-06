#!/bin/bash

BASE_DIR="/home/xavierzeng/workspace/data/"
ROOT_PATH="/home/xavierzeng/workspace/data/test/"
SCRIPT_DIR="/home/xavierzeng/workspace/code/lerobot/lerobot/data_convert_scripts"
BATCH=1
N_SPLIT=$((BATCH * 1))
GPU_LIST=(0)
GPU_NUM=${#GPU_LIST[@]}

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
  RAW_PATH="${BASE_DIR}${SUBDIR_LIST[$idx]}"
  REPO_ID="${REPO_ID_LIST[$idx]}"
  HDF5_DIR="${RAW_PATH}/hdf5"
  MERGED_BASE="${ROOT_PATH}${REPO_ID}"
  MERGED_SPLITS="${MERGED_BASE}/merged1"
  mkdir -p "$MERGED_SPLITS"

  echo "========== 处理 $REPO_ID =========="
  
    # TODO----------------- 新增步骤0: 检测目标hdf5是否包含prompt键值 ------------------
  # echo "🔵 [Step0] 检测 $HDF5_DIR 下的 HDF5 文件是否包含 prompt 键值"
  # python check_hdf5.py "$HDF5_DIR"
  # if [[ $? -ne 0 ]]; then
  #     echo "❌ check_hdf5.py 检测失败，退出"; exit 1
  # fi

  # ------------------ 步骤1: 自动切分 ------------------
  echo "🟠 [Step1] 自动切分 $HDF5_DIR 为 $N_SPLIT 份"
  python split_data_folder.py "$HDF5_DIR" $N_SPLIT
  if [[ $? -ne 0 ]]; then
      echo "❌ split_data_folder.py 失败，退出"; exit 1
  fi

  # ------------------ 步骤2: 动态并发生成子数据集 ------------------
  MERGE_INPUTS=()
  running=0
  for ((j=1; j<=N_SPLIT; j++)); do
      SPLIT_PATH="${HDF5_DIR}/split_${j}"
      REPO_SUBID="${REPO_ID}_split_${j}"
      OUT_ROOT="${MERGED_SPLITS}" 
  done
  
      GPU_IDX=$(( (j-1) % GPU_NUM ))
  
      CUDA_VISIBLE_DEVICES=${GPU_LIST[$GPU_IDX]} \
        python convert_hdf5_to_lerobot.py \
          --description "${TASK_DESCRIPTION_LIST[$idx]}" \
          --raw-path "$SPLIT_PATH" \
          --root-path "$OUT_ROOT" \
          --dataset-repo-id "$REPO_SUBID" \
          --hdf5_compressed True \
          --split False &
      running=$((running+1))
      MERGE_INPUTS+=("$OUT_ROOT")
  
      if [[ $running -ge $BATCH ]]; then
          # 等任意一个子任务结束，空出一个并发位
          wait -n
          running=$((running-1))
      fi
  done
  
  # 等所有剩余进程结束
  wait
  
  echo "✅ 所有 split 处理完成"

  # ------------------ 步骤3: 合并所有 split ------------------
  # 自动查找 merged1 下所有 split 子数据集目录（**仅这一段在运行**）
  MERGE_INPUTS=$(find "$MERGED_SPLITS" -mindepth 1 -maxdepth 1 -type d | paste -sd "," -)
  echo "MERGE_INPUTS: $MERGE_INPUTS"

  echo "🟣 [Step3] 合并 $REPO_ID 的所有子数据集"
  # TODO: 合并粗细指令
  python merge_split_datasets.py \
      --dataset-dirs "$MERGE_INPUTS" \
      --output-dir "$MERGED_BASE" \
      --chunk-size 1000
  echo "✅ 合并完成，结果在：$MERGED_BASE"

  # ------------------ 步骤4: 清理 split 目录 ------------------
  echo "🗑️ 删除所有 split_x 临时目录" 
  for ((j=1; j<=N_SPLIT; j++)); do
      rm -rf "${HDF5_DIR}/split_${j}"
  done

  # ------------------ 步骤5：删除 merged1 目录（先注释） ------------------
  echo "🗑️ 删除 ${MERGED_SPLITS} 目录（包含所有 split 子集）"
  rm -rf "${MERGED_SPLITS}"

  echo "============================"
done

echo "🎉🎉🎉 所有文件夹全部处理完成！"

