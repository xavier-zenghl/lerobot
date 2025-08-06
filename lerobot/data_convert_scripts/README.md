# 数据转换脚本使用指南

本目录包含了将HDF5格式数据转换为LeRobot数据集格式的完整工具链。

## 概述

数据转换流程包含以下步骤：
1. **预处理阶段**：使用 `json_process.py` 为HDF5文件添加prompt标签
2. **转换阶段**：使用转换脚本将HDF5数据转换为LeRobot格式
   - 单进程转换：`s1_converter.sh`
   - 多进程转换：`S1_converter_multi_process.sh`

## 目录结构

```
lerobot/
├── lerobot/
│   └── data_convert_scripts/   # 数据转换脚本目录
│       ├── README.md           # 本说明文档
│       ├── s1_converter.sh    # 单进程数据转换脚本
│       ├── S1_converter_multi_process.sh # 多进程数据转换脚本
│       ├── convert_hdf5_to_lerobot.py # 核心转换脚本
│       ├── split_data_folder.py # 数据分割工具
│       └── merge_split_datasets.py # 数据集合并工具
└── datatools/                  # 数据工具目录
    ├── json_process.py         # 预处理脚本：为HDF5添加prompt
    └── ...
```

## 使用步骤

### 步骤1：预处理 - 添加Prompt标签

在使用转换脚本之前，需要先为HDF5文件添加prompt标签，请根据task修改json_process.py：

```bash
cd lerobot/datatools

# 运行预处理脚本
python json_process.py -i /path/to/your/data/directory
```

**参数说明：**
- `-i`: 输入数据目录路径（包含.json标注文件）

**功能说明：**
- 读取JSON标注文件中的动作信息
- 将动作信息转换为自然语言prompt
- 将prompt信息添加到对应的HDF5文件中

### 步骤2：配置转换脚本

根据你的需求选择单进程或多进程转换方式：

#### 单进程转换 (`s1_converter.sh`)

适用于小规模数据集或调试阶段：


**主要配置项：**
```bash
BASE_DIR="/home/xavierzeng/workspace/data/"          # 原始数据根目录
ROOT_PATH="/home/xavierzeng/workspace/data/test/"    # 输出根目录
SCRIPT_DIR="/home/xavierzeng/workspace/code/lerobot/lerobot/data_convert_scripts"
GPU_IDX=0                                            # 使用的GPU索引

# 数据子目录列表
SUBDIR_LIST=(
  "0731_desktop_Storage_S8"
)

# 对应的数据集仓库ID
REPO_ID_LIST=(
  "lerobot_so3_data_30hz"
)

# 任务描述
TASK_DESCRIPTION_LIST=(
  "A robot is positioned in front of the checkout counter, where three different types of items and a shopping bag are placed. Packing in the supermarket."
)
```

#### 多进程转换 (`S1_converter_multi_process.sh`)
***暂不使用，合并task index还在开发中***

适用于大规模数据集，支持并行处理：

**主要配置项：**
```bash
BASE_DIR="/home/xavierzeng/workspace/data/"
ROOT_PATH="/home/xavierzeng/workspace/data/test/"
SCRIPT_DIR="/home/xavierzeng/workspace/code/lerobot/lerobot/data_convert_scripts"
BATCH=1                                              # 批处理大小
N_SPLIT=$((BATCH * 1))                              # 数据分割份数
GPU_LIST=(0)                                         # 可用的GPU列表
```

### 步骤3：执行转换

#### 单进程转换
```bash
# 执行单进程转换
cd lerobot/lerobot/data_convert_scripts
bash s1_converter.sh
```

#### 多进程转换
```bash
# 执行多进程转换
cd lerobot/lerobot/data_convert_scripts
bash S1_converter_multi_process.sh
```
