#!/bin/bash
export PYTHONWARNINGS="ignore:The video decoding and encoding capabilities of torchvision are deprecated:UserWarning"
source /home/xavierzeng/code/miniconda3/bin/activate
conda activate lerobot_data

# task_id 列表，可以根据实际情况填入

# task_ids=("327" "352" "354" "357" "358" "363" "428") 

# 定义源路径和目标路径
source_path="/home/xavierzeng/mnt/nas_data/temp-data/Agibot_data_raw/observations"
destination_path="/home/xavierzeng/data/agibot/observations"

# 循环遍历 task_id 列表
for task_id in "${task_ids[@]}"
do
    # 构建源路径和目标路径
    src_dir="$source_path/$task_id"
    dest_dir="$destination_path/$task_id"

    # 检查源目录是否存在
    if [ -d "$src_dir" ]; then
        echo "Copying from $src_dir to $dest_dir"
        
        # 创建目标目录
        mkdir -p "$dest_dir"
        
        # 使用 cp 命令复制源目录到目标目录
        cp -r "$src_dir"/* "$dest_dir"
        
        # # 进入目标路径解压所有 tar 文件
        cd "$dest_dir" || exit
        for tar_file in *.tar; do
            if [ -f "$tar_file" ]; then
                echo "Extracting $tar_file..."
                tar -xf "$tar_file"  # 解压 tar 文件
                rm -f "$tar_file"    # 删除解压后的 tar 文件
            fi
        done

        # 使用小写的 task_id 变量
        cd /home/xavierzeng/code/dataset_utils/agibot_dataset_utils
        python convert_agibot_to_so3_lerobot.py --src_path /home/xavierzeng/data/agibot --task_id "$task_id" --tgt_path /home/xavierzeng/data/lerobot_datasets/agibot
        
        # 删除目标目录
        rm -rf "$dest_dir"

        echo "Task $task_id completed."
    else
        echo "Source directory $src_dir does not exist."
    fi
done

echo "All tasks completed."