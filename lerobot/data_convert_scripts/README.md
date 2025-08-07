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
```
    ### 细指令 修改这一部分适应数据集
    subject = label['attributes']['Arm']
    subject = subject[0] if len(subject) > 0 else ''
    object = label['attributes']['Object']
    object = object[0] if len(object) > 0 else ''
    target = label['attributes']['Action']
    target = target[0] if len(target) > 0 else ''
    # target = label['attributes']['Target'][0]

    prompt = TARGET_DICT[target].format(preprocess_object(object))
    
    if subject != '':
        prompt = 'the ' + subject.replace("_", " ")  + ' ' + prompt
            
    ### 细指令 修改这一部分适应数据集

    prompt = prompt.replace('_', ' ')
    clean_labels = {
            'frame': int(label['time'] * 30),
            'prompt': prompt,
            'valid': True
    }
    frames.append(clean_labels)

label_dict[src_file] = frames

### 粗指令 修改这一部分适应数据集
coarse_frames = []
for frame in frames:
    if not frame['valid']:
        coarse_labels = {
            'frame': frame['frame'],
            'prompt': 'None',
            'valid': False
        }
        coarse_frames.append(coarse_labels)
        continue

    ### 方案一，当前任务使用统一的粗指令填充
    coarse_prompt = 'A robot is positioned in front of the checkout counter, where three different types of items and a shopping bag are placed. Packing in the supermarket.'
    coarse_labels = {
        'frame': frame['frame'],
        'prompt': coarse_prompt,
        'valid': True
    }
    coarse_frames.append(coarse_labels)

    ### 方案二，根据当前frame的prompt生成粗指令，请自行填补
    
    ###
### 粗指令 修改这一部分适应数据集
```

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

## 更新日志

### 最新更新

1. **明确并统一state、action各个维度的定义**：已将state和action的维度定义写入`info.json`文件中，确保数据格式的一致性。

2. **优化指令处理方式**：不再对细指令进行视频切分，粗指令和细指令都保存在`task_index`中：
   - `task_index[0]`：粗指令
   - `task_index[1]`：细指令

3. **待办事项**：
   - [x] 更新读取task规则
   - [x] 更新action padding方式
   - [ ] 更新合并数据集
