import os
import sys
import h5py
import numpy as np
import tempfile
import cv2
from scipy.spatial.transform import Rotation as R
from typing import List, Union, Any
from datetime import datetime
from pathlib import Path
import pandas as pd

import functools

import numcodecs
import simplejpeg
from tqdm import tqdm
import json
import shutil
from itertools import dropwhile


def time_compare(src_path: Union[str, Path]):
    if isinstance(src_path, str):
        src_path = Path(src_path)
    # 定义目标时间点（精确到分钟）
    target_time = datetime(2025, 1, 27, 17, 30)  # 2023年10月1日 14:30

    # 遍历文件路径
    if src_path.exists():  # 检查文件是否存在
        # 获取文件的最后修改时间
        modify_timestamp = os.path.getmtime(src_path)
        
        # 将时间戳转换为 datetime 对象
        modify_time = datetime.fromtimestamp(modify_timestamp)
        
        # 将秒和微秒部分设置为 0，精确到分钟
        modify_time = modify_time.replace(second=0, microsecond=0)
        
        # print(modify_time)
        # print(target_time)
        # 比较文件的修改时间和目标时间
        if modify_time > target_time:
            return True
        else:
            return False
    else:
        print(f"文件 {src_path} 不存在")
        return
        
def hdf5_print(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"Dataset: {name}")
        print(f"Shape: {obj.shape}")
    elif isinstance(obj, h5py.Group):
        print(f"Group: {name}")


def xyzquat_to_T44(quaternion):
    """
    将四元数转换为 4x4 变换矩阵
    :param quaternion: 包含位置和四元数信息的数组 (7,)
    :return: 4x4 变换矩阵
    """
    position = quaternion[:3]
    qx, qy, qz, qw = quaternion[3:]
    
    # 四元数转换为旋转矩阵
    R = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy)],
        [2 * (qx * qy + qw * qz), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qw * qx)],
        [2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx**2 + qy**2)]
    ])
    
    # 构建 4x4 变换矩阵
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = position
    
    return T

def merge_txt(log_list: List[str], out_path: str) -> None:
    with open(out_path, 'w') as f_w:
        for f in log_list:
            try:
                with open(f, 'r') as f_r:
                    for l in f_r:
                        f_w.write(l)
            except:
                print(f'fail to merge log file: {f}')
                pass

def quat_to_rpy(quaternions):
    """
    将四元数转换为 RPY（Roll-Pitch-Yaw）欧拉角
    :param quaternions: (n, 4) 的四元数数组，格式为 [x, y, z, w]
    :return: (n, 3) 的 RPY 欧拉角数组，单位为弧度
    """
    # 创建 Rotation 对象
    rotation = R.from_quat(quaternions)
    # 将四元数转换为 RPY 欧拉角（顺序为 'xyz'，即 Roll-Pitch-Yaw）
    rpy = rotation.as_euler('xyz', degrees=False)
    return rpy

def xyz_quat_to_xyz_rpy(data):
    """
    将 (n, 7) 的 xyz + 四元数数据转换为 (n, 6) 的 xyz + RPY 数据
    :param data: (n, 7) 的数组，格式为 [x, y, z, qx, qy, qz, qw]
    :return: (n, 6) 的数组，格式为 [x, y, z, roll, pitch, yaw]
    """
    # 提取 xyz 坐标
    xyz = data[:, :3]
    # 提取四元数部分
    quaternions = data[:, 3:]
    # 将四元数转换为 RPY 欧拉角
    rpy = quat_to_rpy(quaternions)
    # 拼接 xyz 和 RPY
    xyz_rpy = np.hstack([xyz, rpy])
    return xyz_rpy

def quat_to_so3(quaternions):
    """
    将四元数转换为 SO(3) 表示（旋转矩阵的前两行）
    :param quaternions: (n, 4) 的四元数数组，格式为 [qx, qy, qz, qw]
    :return: (n, 6) 的 SO(3) 表示数组，每行包含旋转矩阵的前两行
    """
    # 创建 Rotation 对象
    rotation = R.from_quat(quaternions)
    # 将四元数转换为旋转矩阵
    rotation_matrices = rotation.as_matrix()
    # 提取旋转矩阵的前两行
    so3 = rotation_matrices[:, :2, :].reshape(rotation_matrices.shape[0], -1)
    return so3

def xyz_quat_to_xyz_so3(data):
    """
    将 (n, 7) 的 xyz + 四元数数据转换为 (n, 9) 的 xyz + SO(3) 数据
    :param data: (n, 7) 的数组，格式为 [x, y, z, qx, qy, qz, qw]
    :return: (n, 9) 的数组，格式为 [x, y, z, r11, r12, r13, r21, r22, r23]
    """
    # 提取 xyz 坐标
    xyz = data[:, :3]
    # 提取四元数部分
    quaternions = data[:, 3:]
    # 将四元数转换为 SO(3) 表示
    so3 = quat_to_so3(quaternions)
    # 拼接 xyz 和 SO(3) 表示
    xyz_so3 = np.hstack([xyz, so3])
    return xyz_so3

def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    x2, y2, z2, w2 = x * x, y * y, z * z, w * w
    xy, xz, xw, yz, yw, zw = x * y, x * z, x * w, y * z, y * w, z * w
    return np.array(
        [
            [w2 + x2 - y2 - z2, 2 * (xy - zw), 2 * (xz + yw)],
            [2 * (xy + zw), w2 - x2 + y2 - z2, 2 * (yz - xw)],
            [2 * (xz - yw), 2 * (yz + xw), w2 - x2 - y2 + z2],
        ]
    )

def xyz_quat_to_so3(xyz_quat):
    x, y, z, qx, qy, qz, qw = xyz_quat
    R = quaternion_to_rotation_matrix([qx, qy, qz, qw])

    return np.hstack([xyz_quat[:3], R[:2].flatten()]) 

def so3_to_rpy(so3):
    """
    将SO(3)表示（旋转矩阵的前两行）转换为RPY欧拉角
    :param so3: (n, 6) 的SO(3)数组，每行包含旋转矩阵的前两行 [r11, r12, r13, r21, r22, r23]
    :return: (n, 3) 的RPY欧拉角数组，单位为弧度
    """
    # 将SO(3)数据重塑为完整的旋转矩阵
    rotation_matrices = np.zeros((so3.shape[0], 3, 3))
    rotation_matrices[:, :2, :] = so3.reshape(-1, 2, 3)
    
    # 计算第三行
    rotation_matrices[:, 2, :] = np.cross(rotation_matrices[:, 0, :], rotation_matrices[:, 1, :])
    
    # 创建Rotation对象并转换为RPY
    rotation = R.from_matrix(rotation_matrices)
    rpy = rotation.as_euler('xyz', degrees=True)
    return rpy

def xyz_so3_to_xyz_rpy(data):
    """
    将 (n, 9) 的 xyz + SO(3) 数据转换为 (n, 6) 的 xyz + RPY 数据
    :param data: (n, 9) 的数组，格式为 [x, y, z, r11, r12, r13, r21, r22, r23]
    :return: (n, 6) 的数组，格式为 [x, y, z, roll, pitch, yaw]
    """
    # 提取 xyz 坐标
    xyz = data[:, :3]
    # 提取 SO(3) 部分
    so3 = data[:, 3:]
    # 将 SO(3) 转换为 RPY
    rpy = so3_to_rpy(so3)
    # 拼接 xyz 和 RPY
    xyz_rpy = np.hstack([xyz, rpy])
    return xyz_rpy

def extract_compress_file(mp4_data: h5py.Dataset):
    frame_list = []
    with tempfile.NamedTemporaryFile(suffix=".mp4") as temp_video:
        temp_video.write(mp4_data[:])  # 写入 MP4 数据
        temp_video.flush()  # 确保数据写入磁盘
        # 打开视频文件并读取帧
        cap = cv2.VideoCapture(temp_video.name)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_list.append(frame)  
        cap.release()
    return np.array(frame_list)


def _assert_shape(arr: np.ndarray, expected_shape: tuple[int | None, ...]):
    assert len(arr.shape) == len(expected_shape), (arr.shape, expected_shape)
    for dim, expected_dim in zip(arr.shape, expected_shape):
        if expected_dim is not None:
            assert dim == expected_dim, (arr.shape, expected_shape)


class JpegCodec(numcodecs.abc.Codec):
    """Codec for JPEG compression.

    Used to encode image chunks as JPEGs. Assumes that chunks are uint8 with shape (1, H, W, 3).
    """

    codec_id = "pi_jpeg"

    def __init__(self, quality: int = 95):
        super().__init__()
        self.quality = quality

    def encode(self, buf):
        _assert_shape(buf, (1, None, None, 3))
        assert buf.dtype == "uint8"
        return simplejpeg.encode_jpeg(buf[0], quality=self.quality)

    def decode(self, buf, out=None):
        return simplejpeg.decode_jpeg(buf, buffer=out)


@functools.cache
def register_codecs():
    """Register the custom codecs."""
    numcodecs.registry.register_codec(JpegCodec)


def create_video_from_multiple_cameras(views, output_path, fps=30):
    """
    将多个摄像头的图像序列合并成一个视频，并在每个视频块的左上角添加自定义文字

    参数:
    view_1, view_2, view_3, view_4, view_5, view_6: 包含图像数据的可迭代对象或None（如h5py.Dataset或zarr.Array）
    output_path: 输出视频文件的路径
    fps: 视频的帧率，默认为 30
    """
    try:
        # 过滤掉None的视角，获取最小帧数
        valid_views = [view for view in views if view is not None]
        min_frames = min(len(view) for view in valid_views)

        # 获取单个图像的尺寸
        height = 0
        width = 0
        for view in valid_views:
            height = max(height, view[0].shape[0])
            width = max(width, view[0].shape[1])
        # sample_image = next(view[0] for view in valid_views)
        # height, width = sample_image.shape[:2]

        # 计算输出视频的尺寸（3x2 网格）
        out_width, out_height = width * 3, height * 2

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))

        # 定义要添加的文字
        texts = ["Base 0", "Base 1", "Base 2", "Base 3", "Wrist 0", "Wrist 1"]

        # 定义文字属性
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color = (255, 255, 255)  # 白色
        font_thickness = 2
        
        # 定义6个视角的位置
        positions = [(0, 0), (width, 0), (0, height), (width, height), (width*2, 0), (width*2, height)]
        
        for i in tqdm(range(min_frames)):
            frame = np.zeros((out_height, out_width, 3), dtype=np.uint8)
            
            for view, pos, text in zip(views, positions, texts):
                x, y = pos
                if view is not None:
                    frame[y:y+view.shape[1], x:x+view.shape[2]] = view[i]
                    
                    # 添加文字
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    text_x = x + 10  # 左边距 10 像素
                    text_y = y + text_size[1] + 10  # 上边距 10 像素
                    cv2.putText(frame, text, (text_x, text_y), font, font_scale, font_color, font_thickness)
                else:
                    # 如果视角为None，保持黑色背景并添加文本说明
                    cv2.putText(frame, f"No {text}", (x + 10, y + height // 2), 
                                font, font_scale, font_color, font_thickness)

            # OpenCV 使用 BGR 颜色空间，如果你的图像是 RGB，需要转换
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            out.write(frame)

        out.release()
    except Exception as e:
        if os.path.exists(output_path):
            os.remove(output_path)
        print(f"Error occurred: {e}, remove {output_path}\n")
        sys.exit(1)


def get_all_files(directory):
    path = Path(directory)
    return [file for file in path.iterdir() if file.is_file()]

def get_all_folders(directory):
    path = Path(directory)
    return [file for file in path.iterdir() if not file.is_file()]

def write_to_txt(file_path, data):
    """
    将数据写入txt文件

    参数:
    file_path: txt文件的路径
    data: 要写入的数据，可以是字符串或字符串列表
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        if isinstance(data, list):
            for line in data:
                file.write(line + '\n')
        else:
            file.write(data)

def all_none(lst):
    """
    判断列表中的所有元素是否都是None

    参数:
    lst: 要判断的列表

    返回:
    如果列表中的所有元素都是None，返回True；否则返回False
    """
    return all(x is None for x in lst)


def binary_from_gripper(gripper: np.ndarray) -> Any:
    is_open = True
    state = np.zeros_like(gripper)
    # 遍历序列，记录夹爪角度超过阈值的时刻索引
    for i in range(len(state)):
        if is_open and gripper[i] > 70:
            # 当超过阈值且is_open为True时，记录开始变化的索引
            # is_open = False
            state[i] = 1
        elif not is_open and gripper[i] <= 20:
            # 当已经开始变化且值再次低于等于阈值时，记录变回初始状态的索引
            is_open = True

    return state

def detect_action(left_binary: np.ndarray, right_binary: np.ndarray) -> List[dict]:
    is_grab = False
    cur = 0
    action = []

    while(cur < len(left_binary) - 1):
        if right_binary[cur] == right_binary[cur + 1] and left_binary[cur] == left_binary[cur + 1]:
            cur += 1
            continue

        if right_binary[cur] == 0 and right_binary[cur + 1] == 1 and not is_grab:
            action.append({
                'end': cur + 10,
                'type': 'the right arm pick up the object'
            })
            is_grab = True
        elif left_binary[cur] == 0 and left_binary[cur + 1] == 1 and not is_grab:
            action.append({
                'end': cur + 10,
                'type': 'the left arm pick up the object'
            })
            is_grab = True
        elif right_binary[cur] == 1 and right_binary[cur + 1] == 0 and left_binary[cur] == 1:
            action.append({
                'end': cur + 10,
                'type': 'the right arm deliver the object to the left arm'
            })
        elif left_binary[cur] == 1 and left_binary[cur + 1] == 0 and right_binary[cur] == 1:
            action.append({
                'end': cur + 10,
                'type': 'the left arm deliver the object to the right arm'
            })
        elif right_binary[cur] == 1 and right_binary[cur + 1] == 0 and left_binary[cur] == 0:
            action.append({
                'end': cur + 10,
                'type': 'the right arm put down the object to the trash bin'
            })
            is_grab = False
        elif left_binary[cur] == 1 and left_binary[cur + 1] == 0 and right_binary[cur] == 0:
            action.append({
                'end': cur + 10,
                'type': 'the left arm put down the object to the container'
            })
            is_grab = False

        cur += 1


    return action

def detect_action_pick_only(left_binary: np.ndarray, right_binary: np.ndarray) -> List[dict]:
    action = []

    if right_binary[4] == 1:
        action.append({
            'end': len(right_binary),
            'type': 'the right arm pick up the object'
        })
    elif left_binary[4] == 1:
        action.append({
            'end': len(left_binary),
            'type': 'the left arm pick up the object'
        })

    return action

def analyse_traj(data_list: List[Path], log_file: Path) -> None:
    csv_path = list(data_list[0].parent.rglob('*.csv'))[0]
    prompt_list = pd.read_csv(csv_path)
    with open(log_file, 'w') as f:
        for data in tqdm(data_list):
            episode_idx = data.stem.split('_')[-1]
            text_matched = prompt_list[prompt_list["idx"]==int(episode_idx)]
            object = text_matched['instruction'].values[0]
            # try:
            with h5py.File(data, 'r+') as src:
                # print(data.name)
                # 获取左右臂的位姿和夹爪角度
                right_gripper = np.array(src['joints_dict/joints_position_command'])[:, 21]
                left_gripper = np.array(src['joints_dict/joints_position_command'])[:, 13]
                frame_size = len(right_gripper)
                # 遍历序列，记录夹爪角度超过阈值的时刻索引
                left_binary = binary_from_gripper(left_gripper)
                right_binary = binary_from_gripper(right_gripper)
                actions = detect_action_pick_only(left_binary, right_binary)
                prompts = []
                last = 0
                for action in actions:
                    duration = action['end'] - last
                    if duration < 30:
                        continue
                    if action['end'] > frame_size:
                        action['end'] = frame_size
                    prompts += (action['end'] - last) * [action['type'].replace('object', object)]
                    last = action['end']
                prompts += ['None'] * (frame_size - last)
                assert len(prompts) == frame_size, f"len(prompts) != frame_size, {len(prompts)} != {frame_size}"
                # print(prompts)
                if 'prompt' in src:
                    del src['prompt']
                dt = h5py.string_dtype(encoding="utf-8")
                src.create_dataset('prompt', data=prompts, dtype=dt)
                f.write(f"{data.name}: " + prompts[0] + ", " + prompts[-1] + '\n')

        # except:
        #     os.remove(data)
        #     print(f"fail to analyse trajactory, please check the file: {data}, delete it.")
        #     continue


def transform_annotation_to_instruction(data_Path: Path, out_dir: Path) -> dict:
    with open(data_Path, 'r') as f_r:
        label_dict = {}
        annotations = json.load(f_r)
        for labels in annotations:
            frames = []
            src_file = labels['fileName'].replace('mp4', 'hdf5').replace('garbage_collection', '垃圾分拣')
            src_file = Path(out_dir) / src_file
            # print(src_file)

            # try:
            for label in labels['result']['annotations'][0]['result']:
                # print(label)
                if 'attributes' not in label.keys():
                    clean_labels = {
                            'frame': int(label['time'] * 30),
                            'prompt': 'None',
                            'valid': False
                    }
                    continue
                if label['attributes']['Data_Validity'] == 'invalid_data':
                    clean_labels = {
                            'frame': int(label['time'] * 30),
                            'prompt': 'None',
                            'valid': False
                    }
                    frames.append(clean_labels)
                    continue
                
                subject = label['attributes']['Arm'][0]
                object = label['attributes']['Object'][0]
                target = label['attributes']['Target'][0]
                        
                if target == 'grab':
                    prompt = 'pick up the ' + object +' with the ' + subject
                elif target == 'scoop_up':
                    prompt = 'scoop up ' + object +' with the ' + subject
                elif target == 'pour_into_the_bowl':
                    prompt = "pour the "+ object +" into the bowl" + " with the "+ subject
                elif target == 'put_down':
                    prompt = "put down "+ object  + " with the "+ subject    # put down the shovel with the right arm
                elif target == 'shovel_up_again':
                    prompt = 'scoop up ' + object +' with the ' + subject + ' again' # scoop up the object again with the right arm
                    
                else:
                    # if subject == 'left_arm':
                    #     target = 'right_arm'
                    # else:
                    #     target = 'left_arm'
                    # prompt = 'the ' + subject + ' deliver the ' + object + ' to the ' + target
                    raise ValueError(f"Invalid subject: {target}. Expected.")

                prompt = prompt.replace('_', ' ')
                print(prompt)
                clean_labels = {
                        'frame': int(label['time'] * 30),
                        'prompt': prompt,
                        'valid': True
                }
                frames.append(clean_labels)
            
            label_dict[src_file] = frames
            # except Exception as e:
            #     print(src_file)
    
    return label_dict

def add_prompt_hdf5(data_list: List[Path], out_dir: str, log_file: Path | str) -> None:
    labels = {}

    for data in tqdm(data_list):
        labels.update(transform_annotation_to_instruction(data, out_dir))
    # print(labels)

    with open(log_file, 'w') as f:
        for data, label in labels.items():
            prompts = []
            if not data.exists():
                print(f"{data} not exists")
                continue

            with h5py.File(data, 'r+') as src:
                length = src['time'].shape[0]
                if 'prompt' in src.keys():
                    del src['prompt']
                last_frame = 0
                for l in label:
                    valid = l['valid']
                    prompt = l['prompt'] if valid else 'None'
                    prompts += [prompt] * (l['frame'] - last_frame)
                    last_frame = l['frame']
                prompts += ['None'] * (length - last_frame)
                # print(prompts)
                if 'prompt' in src.keys():
                    del src['prompt']
                dt = h5py.string_dtype(encoding="utf-8")
                src.create_dataset('prompt', data=prompts, dtype=dt)     
            f.write(f"{data.name}: " + prompts[0] + ", " + prompts[-1] + '\n')

def num2char(prompt: List[int]) -> str:
    prompt = list(dropwhile(lambda x: x == 32, reversed(prompt)))[::-1]
    string = ''.join(chr(num) for num in prompt)
    return string

def stats_all_prompt(data_list: List[Path]) -> Any:
    labels = dict()
    for data in tqdm(data_list):
        df = pd.read_parquet(data)
        prompt = df['prompt'].to_list()[0]
        prompt = list(dropwhile(lambda x: x == 32, reversed(prompt)))[::-1]
        string = ''.join(chr(num) for num in prompt)
        
        if string.split('\x00')[0] in labels.keys():
            labels[string.split('\x00')[0]].append(str(data))
        else:
            labels[string.split('\x00')[0]] = [str(data)]
    
    return labels

def filter_lerobot(root_dir: Path, destination_root_dir: Path, selected_ls: List[Path]):
    info_path = root_dir / 'meta' / 'info.json'
    tasks_path = root_dir / 'meta' / 'tasks.jsonl'
    
    destination_root_dir.mkdir(parents=True, exist_ok=True)

    meta_dir_des = destination_root_dir / 'meta'
    meta_dir_des.mkdir(parents=True, exist_ok=True)

    episodes_path_des = meta_dir_des / 'episodes.jsonl'
    episodes_path_des.touch()

    info_path_des = meta_dir_des / info_path.name
    tasks_path_des = meta_dir_des / tasks_path.name

    frame_num = 0


    num = 0
    shutil.copy(tasks_path, tasks_path_des)
    shutil.copy(info_path, info_path_des)

    video_ls = get_all_folders(root_dir / 'videos' / 'chunk-000')


    with open(episodes_path_des, 'w') as f:
        for path in tqdm(selected_ls):
            path = Path(path)
            df = pd.read_parquet(path)
            prompt = df['prompt'].to_list()[0]
            prompt_str = num2char(prompt)

            frame_num += df.shape[0]

            chunk_num = num // 1000

            for vpath in video_ls:
                mp4_path = vpath.parent.parent / path.parent.name / vpath.name / path.with_suffix('.mp4').name
                mp4_path_des = destination_root_dir / 'videos' / f'chunk-{chunk_num:03d}' / vpath.name / f'episode_{num:06d}.mp4'
                mp4_path_des.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(mp4_path, mp4_path_des)

            data_path = destination_root_dir / 'data' / f'chunk-{chunk_num:03d}' / f'episode_{num:06d}.parquet'
            data_path.parent.mkdir(parents=True, exist_ok=True)
            
            shutil.copy(path, data_path)
            
            episode_info = {
                "episode_index": num,
                "tasks": [prompt_str],
                "length": df.shape[0],
            }
            f.write(json.dumps(episode_info) + '\n')
            num += 1

    with open(info_path_des, 'r', encoding='utf-8') as f:
        data = json.load(f)

        data['total_episodes'] = num
        data['total_frames'] = frame_num
        data['total_videos'] = len(video_ls) * num
        data['total_chunks'] = chunk_num + 1
        data['splits']['train'] = f'0:{num}'

        with open(info_path_des, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4)