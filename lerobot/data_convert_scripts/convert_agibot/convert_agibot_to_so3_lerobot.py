""" 
This project is built upon the open-source project 🤗 LeRobot: https://github.com/huggingface/lerobot 

We are grateful to the LeRobot team for their outstanding work and their contributions to the community. 

If you find this project useful, please also consider supporting and exploring LeRobot. 
"""

import os
import json
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from typing import Callable
from functools import partial
from math import ceil
from copy import deepcopy
from scipy.spatial.transform import Rotation as R

import h5py
import torch
import einops
import numpy as np
from PIL import Image
from tqdm import tqdm
from pprint import pformat
import jsonlines
from tqdm.contrib.concurrent import process_map
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import (
    STATS_PATH,
    TASKS_PATH,
    append_jsonlines,
    check_timestamps_sync,
    get_episode_data_index,
    serialize_dict,
    write_json,
)
from datatools.features_utils import DATASET_FEATURES

HEAD_COLOR = "head_color.mp4"
HAND_LEFT_COLOR = "hand_left_color.mp4"
HAND_RIGHT_COLOR = "hand_right_color.mp4"

DEFAULT_IMAGE_PATH = (
    "images/{image_key}/episode_{episode_index:06d}/frame_{frame_index:06d}.jpg"
)


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

def rotate_quaternion(q):
    q = np.asarray(q)
    rotation_q = np.array([0.0, -0.707, 0.707, 0.0])

    x1, y1, z1, w1 = q[:,0], q[:,1], q[:,2], q[:,3]
    x2, y2, z2, w2 = rotation_q

    x = x1 * w2 + w1 * x2 + y1 * z2 - z1 * y2 # x
    y = y1 * w2 + w1 * y2 + z1 * x2 - x1 * z2 # y
    z = z1 * w2 + w1 * z2 + x1 * y2 - y1 * x2 # z
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2 # w

    return np.stack([x, y, z, w], axis=1)

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
    # 四元数转到Astribot机器人夹爪坐标轴方向对齐
    rotate_quaternions = rotate_quaternion(quaternions)
    # 将四元数转换为 SO(3) 表示
    so3 = quat_to_so3(rotate_quaternions)
    # 拼接 xyz 和 SO(3) 表示
    xyz_so3 = np.hstack([xyz, so3])
    return xyz_so3

def get_stats_einops_patterns(dataset, num_workers=0):
    """These einops patterns will be used to aggregate batches and compute statistics.

    Note: We assume the images are in channel first format
    """

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=2,
        shuffle=False,
    )
    batch = next(iter(dataloader))

    stats_patterns = {}

    for key in dataset.features:
        if "images_dict" in key:
            continue
        # sanity check that tensors are not float64
        assert batch[key].dtype != torch.float64

        # if isinstance(feats_type, (VideoFrame, Image)):
        if key in dataset.meta.camera_keys:
            # sanity check that images are channel first
            _, c, h, w = batch[key].shape
            assert (
                c < h and c < w
            ), f"expect channel first images, but instead {batch[key].shape}"
            assert (
                batch[key].dtype == torch.float32
            ), f"expect torch.float32, but instead {batch[key].dtype=}"
            # assert batch[key].max() <= 1, f"expect pixels lower than 1, but instead {batch[key].max()=}"
            # assert batch[key].min() >= 0, f"expect pixels greater than 1, but instead {batch[key].min()=}"
            stats_patterns[key] = "b c h w -> c 1 1"
        elif batch[key].ndim == 2:
            stats_patterns[key] = "b c -> c "
        elif batch[key].ndim == 1:
            stats_patterns[key] = "b -> 1"
        else:
            raise ValueError(f"{key}, {batch[key].shape}")

    return stats_patterns

def get_features_dict():
    features_dict = {
        "images_dict.head.rgb": {
            "dtype": "video",
            "shape": [3, 480, 640],
            "names": ["channel", "height", "width"],
            "video_info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "images_dict.left.rgb": {
            "dtype": "video",
            "shape": [3, 480, 640],
            "names": ["channel", "height", "width"],
            "video_info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        },
        "images_dict.right.rgb": {
            "dtype": "video",
            "shape": [3, 480, 640],
            "names": ["channel", "height", "width"],
            "video_info": {
                "video.fps": 30.0,
                "video.height": 480,
                "video.width": 640,
                "video.channels": 3,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "has_audio": False,
            },
        }
    }
    for key in DATASET_FEATURES["agibot"]:
        features_dict[key] = {
            "dtype": "float64",
            "shape": (len(DATASET_FEATURES["agibot"][key]),),
            "names": DATASET_FEATURES["agibot"][key],
        }

    return features_dict

def compute_stats(dataset, batch_size=64, num_workers=20, max_num_samples=None):
    """Compute mean/std and min/max statistics of all data keys in a LeRobotDataset."""
    if max_num_samples is None:
        max_num_samples = len(dataset)

    # 检查CUDA是否可用
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    # for more info on why we need to set the same number of workers, see `load_from_videos`
    stats_patterns = get_stats_einops_patterns(dataset, num_workers)

    # mean and std will be computed incrementally while max and min will track the running value.
    mean, std, max, min = {}, {}, {}, {}
    for key in stats_patterns:
        mean[key] = torch.tensor(0.0).float().to(device)
        std[key] = torch.tensor(0.0).float().to(device)
        max[key] = torch.tensor(-float("inf")).float().to(device)
        min[key] = torch.tensor(float("inf")).float().to(device)

    def create_seeded_dataloader(dataset, batch_size, seed):
        generator = torch.Generator()
        generator.manual_seed(seed)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            generator=generator,
        )
        return dataloader

    # Note: Due to be refactored soon. The point of storing `first_batch` is to make sure we don't get
    # surprises when rerunning the sampler.
    first_batch = None
    running_item_count = 0  # for online mean computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm(
            dataloader,
            total=ceil(max_num_samples / batch_size),
            desc="Compute mean, min, max",
        )
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        if first_batch is None:
            first_batch = deepcopy(batch)
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float().to(device)
            # Numerically stable update step for mean computation.
            batch_mean = einops.reduce(batch[key], pattern, "mean")
            # Hint: to update the mean we need x̄ₙ = (Nₙ₋₁x̄ₙ₋₁ + Bₙxₙ) / Nₙ, where the subscript represents
            # the update step, N is the running item count, B is this batch size, x̄ is the running mean,
            # and x is the current batch mean. Some rearrangement is then required to avoid risking
            # numerical overflow. Another hint: Nₙ₋₁ = Nₙ - Bₙ. Rearrangement yields
            # x̄ₙ = x̄ₙ₋₁ + Bₙ * (xₙ - x̄ₙ₋₁) / Nₙ
            mean[key] = (
                mean[key]
                + this_batch_size * (batch_mean - mean[key]) / running_item_count
            )
            max[key] = torch.maximum(
                max[key], einops.reduce(batch[key], pattern, "max")
            )
            min[key] = torch.minimum(
                min[key], einops.reduce(batch[key], pattern, "min")
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    first_batch_ = None
    running_item_count = 0  # for online std computation
    dataloader = create_seeded_dataloader(dataset, batch_size, seed=1337)
    for i, batch in enumerate(
        tqdm(dataloader, total=ceil(max_num_samples / batch_size), desc="Compute std")
    ):
        this_batch_size = len(batch["index"])
        running_item_count += this_batch_size
        # Sanity check to make sure the batches are still in the same order as before.
        if first_batch_ is None:
            first_batch_ = deepcopy(batch)
            for key in stats_patterns:
                assert torch.equal(first_batch_[key], first_batch[key])
        for key, pattern in stats_patterns.items():
            batch[key] = batch[key].float().to(device)
            # Numerically stable update step for mean computation (where the mean is over squared
            # residuals).See notes in the mean computation loop above.
            batch_std = einops.reduce((batch[key] - mean[key]) ** 2, pattern, "mean")
            std[key] = (
                std[key] + this_batch_size * (batch_std - std[key]) / running_item_count
            )

        if i == ceil(max_num_samples / batch_size) - 1:
            break

    for key in stats_patterns:
        std[key] = torch.sqrt(std[key])

    stats = {}
    for key in stats_patterns:
        stats[key] = {
            "mean": mean[key].cpu(),  # 将结果移回CPU
            "std": std[key].cpu(),    # 将结果移回CPU
            "max": max[key].cpu(),    # 将结果移回CPU
            "min": min[key].cpu(),    # 将结果移回CPU
        }
    return stats


class AgiBotDataset(LeRobotDataset):
    def __init__(
        self,
        repo_id: str,
        root: str | Path | None = None,
        episodes: list[int] | None = None,
        image_transforms: Callable | None = None,
        delta_timestamps: dict[list[float]] | None = None,
        tolerance_s: float = 1e-4,
        download_videos: bool = True,
        local_files_only: bool = False,
        video_backend: str | None = None,
    ):
        super().__init__(
            repo_id=repo_id,
            root=root,
            episodes=episodes,
            image_transforms=image_transforms,
            delta_timestamps=delta_timestamps,
            tolerance_s=tolerance_s,
            download_videos=download_videos,
            local_files_only=local_files_only,
            video_backend=video_backend,
        )

    def save_episode(
        self, task: str, episode_data: dict | None = None, videos: dict | None = None
    ) -> None:
        """
        We rewrite this method to copy mp4 videos to the target position
        """
        if not episode_data:
            episode_buffer = self.episode_buffer

        episode_length = episode_buffer.pop("size")
        episode_index = episode_buffer["episode_index"]
        if episode_index != self.meta.total_episodes:
            # TODO(aliberts): Add option to use existing episode_index
            raise NotImplementedError(
                "You might have manually provided the episode_buffer with an episode_index that doesn't "
                "match the total number of episodes in the dataset. This is not supported for now."
            )

        if episode_length == 0:
            raise ValueError(
                "You must add one or several frames with `add_frame` before calling `add_episode`."
            )

        task_index = self.meta.get_task_index(task)

        buffer_keys = set(episode_buffer.keys())
        feature_keys = set(self.features.keys())
        if buffer_keys != feature_keys:
            print("episode_buffer keys:", sorted(buffer_keys))
            print("self.features keys:", sorted(feature_keys))
            print("Missing in episode_buffer:", feature_keys - buffer_keys)
            print("Extra in episode_buffer:", buffer_keys - feature_keys)
            raise ValueError("Keys in episode_buffer do not match self.features")

        for key, ft in self.features.items():
            if key == "index":
                episode_buffer[key] = np.arange(
                    self.meta.total_frames, self.meta.total_frames + episode_length
                )
            elif key == "episode_index":
                episode_buffer[key] = np.full((episode_length,), episode_index)
            # elif key == "task_index":
            #     episode_buffer[key] = np.full((episode_length,), task_index)
            elif key == "sub_task_index":
                fine_task_index = episode_buffer["task_index"][:, 1]
                sub_task_start = np.zeros(episode_length, dtype=np.int64)
                sub_task_end   = np.zeros(episode_length, dtype=np.int64)

                # 计算每个 frame 在当前细指令中的位置
                current_count = 0
                for i in range(episode_length):
                    if i > 0 and fine_task_index[i] != fine_task_index[i-1]:
                        current_count = 0
                    sub_task_start[i] = current_count
                    current_count += 1

                # 优化后的 O(N) 方法计算每个 frame 到细指令结束的剩余步数
                next_change = episode_length
                for i in range(episode_length - 1, -1, -1):
                    # 如果是最后一帧或下一个帧的任务编号不同，则更新 next_change
                    if i == episode_length - 1 or fine_task_index[i] != fine_task_index[i+1]:
                        next_change = i + 1
                    sub_task_end[i] = next_change - i - 1

                # 将 start/end 堆叠后写入 buffer
                episode_buffer["sub_task_index"] = np.stack([sub_task_start, sub_task_end], axis=1)
            elif ft["dtype"] in ["image", "video"]:
                continue
            elif len(ft["shape"]) == 1 and ft["shape"][0] == 1:
                episode_buffer[key] = np.array(episode_buffer[key], dtype=ft["dtype"])
            elif len(ft["shape"]) == 1 and ft["shape"][0] > 1:
                episode_buffer[key] = np.stack(episode_buffer[key])
            else:
                raise ValueError(key)

        self._wait_image_writer()
        self._save_episode_table(episode_buffer, episode_index)

        self.meta.save_episode(episode_index, episode_length, task, task_index)
        for key in self.meta.video_keys:
            video_path = self.root / self.meta.get_video_file_path(episode_index, key)
            episode_buffer[key] = video_path
            video_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(videos[key], video_path)
        if not episode_data:  # Reset the buffer
            self.episode_buffer = self.create_episode_buffer()
        self.consolidated = False

    def consolidate(
        self, run_compute_stats: bool = False, keep_image_files: bool = False
    ) -> None:
        self.hf_dataset = self.load_hf_dataset()
        self.episode_data_index = get_episode_data_index(
            self.meta.episodes, self.episodes
        )
        check_timestamps_sync(
            self.hf_dataset, self.episode_data_index, self.fps, self.tolerance_s
        )
        if len(self.meta.video_keys) > 0:
            self.meta.write_video_info()

        if not keep_image_files:
            img_dir = self.root / "images"
            if img_dir.is_dir():
                shutil.rmtree(self.root / "images")
        video_files = list(self.root.rglob("*.mp4"))
        assert len(video_files) == self.num_episodes * len(self.meta.video_keys)

        parquet_files = list(self.root.rglob("*.parquet"))
        assert len(parquet_files) == self.num_episodes

        if run_compute_stats:
            self.stop_image_writer()
            self.meta.stats = compute_stats(self)
            serialized_stats = serialize_dict(self.meta.stats)
            write_json(serialized_stats, self.root / STATS_PATH)
            self.consolidated = True
        else:
            logging.warning(
                "Skipping computation of the dataset statistics, dataset is not fully consolidated."
            )

    def add_frame(self, frame: dict, tasks: list[str], coarse_task: str) -> None:
        """
        This function only adds the frame to the episode_buffer. Apart from images — which are written in a
        temporary directory — nothing is written to disk. To save those frames, the 'save_episode()' method
        then needs to be called.
        """
        # TODO(aliberts, rcadene): Add sanity check for the input, check it's numpy or torch,
        # check the dtype and shape matches, etc.

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        frame_index = self.episode_buffer["size"]
        timestamp = (
            frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        )
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)

        for key in frame:
            if key not in self.features:
                raise ValueError(key)
            item = (
                frame[key].numpy()
                if isinstance(frame[key], torch.Tensor)
                else frame[key]
            )
            self.episode_buffer[key].append(item)

        self.episode_buffer["size"] += 1

        task_indexs = []
        coarse_task_index = self.meta.get_task_index(coarse_task)
        task_indexs.append(coarse_task_index)
        if coarse_task_index not in self.meta.tasks:
            self.meta.info["total_tasks"] += 1
            self.meta.tasks[coarse_task_index] = coarse_task
            task_dict = {
                "task_index": coarse_task_index,
                "task": coarse_task,
            }
            append_jsonlines(task_dict, self.root / TASKS_PATH)

        for task in tasks:
            task_index = self.meta.get_task_index(task)
            task_indexs.append(task_index)
            if task_index not in self.meta.tasks:
                self.meta.info["total_tasks"] += 1
                self.meta.tasks[task_index] = task
                task_dict = {
                    "task_index": task_index,
                    "task": task,
                }
                append_jsonlines(task_dict, self.root / TASKS_PATH)
        
        
        self.episode_buffer["task_index"].append(task_indexs)
        

def load_depths(root_dir: str, camera_name: str):
    cam_path = Path(root_dir)
    all_imgs = sorted(list(cam_path.glob(f"{camera_name}*")))
    return [np.array(Image.open(f)).astype(np.float32) / 1000 for f in all_imgs]


def tansform_chassis(position, orientation):
    pos_x, pos_y, pos_z = position[:, 0], position[:, 1], position[:, 2]
    ori_x, ori_y, ori_z, ori_w = orientation[:, 0], orientation[:, 1], orientation[:, 2], orientation[:, 3]
    ori_yaw = 2 * np.arctan2(ori_z, ori_w)
    return np.column_stack([pos_x, pos_y, ori_yaw])

def load_local_dataset(episode_id: int, src_path: str, task_id: int) -> list | None:
    """Load local dataset and return a dict with observations and actions"""

    ob_dir = Path(src_path) / f"observations/{task_id}/{episode_id}"
    proprio_dir = Path(src_path) / f"proprio_stats/{task_id}/{episode_id}"

    try:
        with h5py.File(proprio_dir / "proprio_stats.h5") as f:

            # joint
            state_joint = np.array(f["state/joint/position"])
            state_effector = np.array(f["state/effector/position"])
            state_effector = np.clip(((state_effector - 34) / (123 - 34)) * 100, 0, 100) # 将值从 34-123 转换为 0-100，确保为正值

            state_chassis_orientation = np.array(f["state/robot/orientation"])
            state_chassis_position = np.array(f["state/robot/position"])
            state_head_position = np.array(f["state/head/position"])

            action_joint = np.array(f["action/joint/position"])
            action_effector = np.array(f["action/effector/position"])

            action_effector = np.clip(action_effector * 100, 0, 100)  # 将值从 0-1 转换为 0-100，确保为正值
            action_chassis_velocity = np.array(f["action/robot/velocity"])
            action_head_position = np.array(f["action/head/position"])

            # eef
            state_eef_position = np.array(f["state/end/position"])
            state_eef_orientation = np.array(f["state/end/orientation"])
            action_eef_position = np.array(f["action/end/position"])
            action_eef_orientation = np.array(f["action/end/orientation"])

        # 创建25维的joints_position_state数组
        joints_position_state = np.zeros((len(state_joint), 25), dtype=np.float64)
        
        # 设置chassis的值

        if state_chassis_position.shape[0] == action_joint.shape[0]:
            chassis_state = tansform_chassis(state_chassis_position, state_chassis_orientation)
            joints_position_state[:, 0:3] = chassis_state  # chassis_state (x, y, yaw)
        
        # 设置state_joint的值
        joints_position_state[:, 7:14] = state_joint[:, 0:7]  # state_joint[0,7]
        joints_position_state[:, 15:22] = state_joint[:, 7:14]  # state_joint[7,14]
        
        # 设置state_effector的值
        joints_position_state[:, 14] = state_effector[:, 0]  # state_effector[0]
        joints_position_state[:, 22] = state_effector[:, 1]  # state_effector[1]

        # 设置state_head的值
        joints_position_state[:, 23:25] = state_head_position  # state_head_position (yaw, pitch)

        # 创建25维的joints_position_command数组
        joints_position_command = np.zeros((len(action_joint), 25), dtype=np.float64)
        
        # 设置chassis的值
        if action_chassis_velocity.shape[0] == action_joint.shape[0]:
            joints_position_command[:, 0:2] = action_chassis_velocity[:]  # chassis_action (x, yaw)
        # 设置action_joint的值
        joints_position_command[:, 7:14] = action_joint[:, 0:7]  # action_joint[0,7]
        joints_position_command[:, 15:22] = action_joint[:, 7:14]  # action_joint[7,14]
        
        # 设置action_effector的值
        joints_position_command[:, 14] = action_effector[:, 0]  # action_effector[0]
        joints_position_command[:, 22] = action_effector[:, 1]  # action_effector[1]

        # 设置action_head的值
        joints_position_command[:, 23:25] = action_head_position  # action_head_position (yaw, pitch)

        # joints_position_command第一维删除,最后一维复制
        last_joints_command = joints_position_command[-1].copy()
        joints_position_command = np.vstack([
            joints_position_command[1:],  # 删除第一个元素
            last_joints_command   # 添加最后一个元素
        ])
        

        # 处理state_eef数据
        state_eef_0 = np.hstack([state_eef_position[:, 0, :], state_eef_orientation[:, 0, :]])
        state_eef_1 = np.hstack([state_eef_position[:, 1, :], state_eef_orientation[:, 1, :]])
        
        # 转换为xyz_so3格式
        state_eef_0_so3 = xyz_quat_to_xyz_so3(state_eef_0)
        state_eef_1_so3 = xyz_quat_to_xyz_so3(state_eef_1)
        
        # 创建31维的cartesian_pose_state数组
        cartesian_pose_state = np.zeros((len(state_joint), 34), dtype=np.float64)

        if state_chassis_position.shape[0] == action_joint.shape[0]:
            cartesian_pose_state[:, -3:] = chassis_state
        
        # 设置state_eef的值
        cartesian_pose_state[:, 9:18] = state_eef_0_so3  # state_eef[0]
        cartesian_pose_state[:, 19:28] = state_eef_1_so3  # state_eef[1]
        
        # 设置state_effector的值
        cartesian_pose_state[:, 18] = state_effector[:, 0]  # state_effector[0]
        cartesian_pose_state[:, 28] = state_effector[:, 1]  # state_effector[1]

        # 设置state_head的值
        cartesian_pose_state[:, 29:31] = state_head_position  # state_head_position (yaw, pitch)

        # 处理action_eef数据
        # 由于数据原因 从state复制
        # action_eef_0 = np.hstack([action_eef_position[:, 0, :], action_eef_orientation[:, 0, :]])
        # action_eef_1 = np.hstack([action_eef_position[:, 1, :], action_eef_orientation[:, 1, :]])
        
        # 转换为xyz_so3格式
        # action_eef_0_so3 = xyz_quat_to_xyz_so3(action_eef_0)
        # action_eef_1_so3 = xyz_quat_to_xyz_so3(action_eef_1)
        
        # 创建31维的cartesian_pose_command数组
        cartesian_pose_command = np.zeros((len(action_joint), 34), dtype=np.float64)

        if action_chassis_velocity.shape[0] == action_joint.shape[0]:
            cartesian_pose_command[:, -3:-1] = action_chassis_velocity[:]  # chassis_action (x, yaw)
        
        # 设置action_eef的值
        # cartesian_pose_command[:, 9:18] = action_eef_0_so3  # action_eef[0]
        # cartesian_pose_command[:, 19:28] = action_eef_1_so3  # action_eef[1]
        cartesian_pose_command[:, 9:18] = state_eef_0_so3.copy()  # action_eef[0]
        cartesian_pose_command[:, 19:28] = state_eef_1_so3.copy()  # action_eef[1]
        
        # 设置action_effector的值
        cartesian_pose_command[:, 18] = action_effector[:, 0]  # action_effector[0]
        cartesian_pose_command[:, 28] = action_effector[:, 1]  # action_effector[1]

        # 设置action_head的值
        cartesian_pose_command[:, 29:31] = action_head_position  # action_head_position (yaw, pitch)

        # cartesian_pose_command第一维删除,最后一维复制
        last_so3_command = cartesian_pose_command[-1].copy()
        cartesian_pose_command = np.vstack([
            cartesian_pose_command[1:],  # 删除第一个元素
            last_so3_command  # 添加最后一个元素
        ])
    
    except Exception as e:
        logging.error(f"load_local_dataset failed: {e}")
        return None
    
    # 获取任务信息
    task_json_path = Path(src_path) / f"task_info/task_{task_id}.json"
    if not task_json_path.exists():
        print(f"Task info file not found: {task_json_path}")
        return None
        
    with open(task_json_path, "r") as f:
        task_info = json.load(f)
    
    # 查找当前episode的信息
    episode_info = None
    for info in task_info:
        if info.get("episode_id") == episode_id:
            episode_info = info
            break
    
    if episode_info is None or "label_info" not in episode_info or "action_config" not in episode_info["label_info"]:
        print(f"No action config found for episode {episode_id}")
        return None
    
    # 根据action_config切分episode
    action_configs = episode_info["label_info"]["action_config"]
    all_frames = []
    all_action_texts = []
    
    # 原始视频路径
    v_path = ob_dir / "videos"
    original_videos = {
        "images_dict.head.rgb": v_path / HEAD_COLOR,
        "images_dict.left.rgb": v_path / HAND_LEFT_COLOR,
        "images_dict.right.rgb": v_path / HAND_RIGHT_COLOR,
    }

    # 粗指令
    task_name = episode_info['init_scene_text'] + " " + episode_info['task_name']
    
    seg_start_frame = action_configs[0]["start_frame"]
    seg_end_frame = action_configs[-1]["end_frame"]

    all_frames = [
        {
            "joints_dict.joints_position_state": joints_position_state[j],
            "joints_dict.joints_position_command": joints_position_command[j],
            "cartesian_so3_dict.cartesian_pose_state": cartesian_pose_state[j],
            "cartesian_so3_dict.cartesian_pose_command": cartesian_pose_command[j],
        }
        for j in range(seg_start_frame, seg_end_frame)
    ]

    for i, action_config in enumerate(action_configs):
        start_frame = action_config["start_frame"]
        end_frame = action_config["end_frame"]

        # 确保frame索引在有效范围内
        start_frame = max(0, start_frame)
        end_frame = min(len(joints_position_state), end_frame)
        
        # 细指令
        action_text = [[action_config["action_text"]]] * (end_frame - start_frame)

        if start_frame >= end_frame:
            print(f"Invalid frame range: {start_frame} - {end_frame}")
            return None
            
        # 提取该action对应的frames
        all_action_texts.extend(action_text)
        
    # 为每个episode创建对应的视频片段
    sub_videos = {}
    for video_key, video_path in original_videos.items():
        # 创建输出目录
        output_dir = v_path / f"segments"
        output_dir.mkdir(exist_ok=True)
        
        # 创建输出文件路径
        output_path = output_dir / f"{video_path.stem}_segment.mp4"
        
        # 检查视频片段是否已经存在
        if output_path.exists():
            print(f"Video segment already exists: {output_path}")
            sub_videos[video_key] = output_path
        else:
            # 截取视频片段
            print(f"Extracting video segment: {output_path}")
            success = extract_video_segment(video_path, output_path, seg_start_frame, seg_end_frame)
            if success:
                sub_videos[video_key] = output_path
            else:
                # 如果截取失败，使用原始视频
                print(f"Failed to extract video segment, using original video: {video_path}")
                sub_videos[video_key] = video_path
    
    
    return all_frames, sub_videos, task_name, all_action_texts


def get_task_instruction(task_json_path: str, task_id: int) -> dict:
    """Get task language instruction"""
    with open(task_json_path, "r") as f:
        task_info = json.load(f)
    task_name = task_info[0]["task_name"]
    task_init_scene = task_info[0]["init_scene_text"]
    task_instruction = f"{task_name}.{task_init_scene}"
    print(f"Get Task Instruction <{task_instruction}>")
    task_name = task_name.replace(" ", "_")
    task_desc = task_name + f"_{task_id}"
    print(f"Get Task Desc <{task_desc}>")
    return task_desc



def main(
    src_path: str,
    tgt_path: str,
    task_id: int,
    task_info_json: str,
    debug: bool = False,
):
    task_desc = get_task_instruction(task_info_json, task_id)
    task_desc = task_desc.replace(".", "")
    repo_id = f"{task_desc}/lerobot_so3_data_30hz"

    features = get_features_dict()

    dataset = AgiBotDataset.create(
        repo_id=repo_id,
        root=f"{tgt_path}/{repo_id}",
        fps=30,
        robot_type="agibot-go1",
        features=features,
        tolerance_s=0.06
    )

    all_subdir = sorted(
        [
            f.as_posix()
            for f in Path(src_path).glob(f"observations/{task_id}/*")
            if f.is_dir()
        ]
    )

    if debug:
        all_subdir = all_subdir[:2]

    # Get all episode id
    all_subdir_eids = [int(Path(path).name) for path in all_subdir]

    if debug:
        raw_datasets_before_filter = [
            load_local_dataset(subdir, src_path=src_path, task_id=task_id)
            for subdir in tqdm(all_subdir_eids)
        ]
    else:
        raw_datasets_before_filter = process_map(
            partial(load_local_dataset, src_path=src_path, task_id=task_id),
            all_subdir_eids,
            max_workers=os.cpu_count() // 2,
            desc="Generating local dataset",
        )


    # remove the None result from the raw_datasets
    raw_datasets = [
        dataset for dataset in raw_datasets_before_filter if dataset is not None
    ]

    # remove the None result from the subdirs
    all_subdir_eids = [
        eid
        for eid, dataset in zip(all_subdir_eids, raw_datasets_before_filter)
        if dataset is not None
    ]
    
    # 为每个episode创建描述
    episode_desc = []
    for raw_dataset in raw_datasets:
        episode_desc.append(raw_dataset[2])
    
    print(f"Total episodes after splitting: {len(episode_desc)}")
    
    # 处理所有子episode
    episode_idx = 0
    for raw_dataset in tqdm(raw_datasets, desc="Generating dataset from raw datasets"):
        # 为每个episode添加帧
        for frame, task_list in tqdm(zip(raw_dataset[0], raw_dataset[3]), desc=f"Generating dataset from episode {episode_idx}"):
            # import ipdb; ipdb.set_trace()
            dataset.add_frame(frame, task_list, raw_dataset[2])
        
        # 保存episode，使用对应的视频片段
        dataset.save_episode(task=raw_dataset[2], videos=raw_dataset[1])
        episode_idx += 1
            

    
    dataset.consolidate(run_compute_stats=False)


# 添加一个函数来截取视频片段
def extract_video_segment(video_path, output_path, start_frame, end_frame, fps=30):
    """
    使用ffmpeg截取视频片段
    
    Args:
        video_path: 输入视频路径
        output_path: 输出视频路径
        start_frame: 起始帧索引
        end_frame: 结束帧索引
        fps: 视频帧率，默认为30
    """
    # 计算开始和结束时间（秒）
    start_time = start_frame / fps
    duration = (end_frame - start_frame) / fps
    
    # 构建ffmpeg命令
    cmd = [
        "ffmpeg", "-y",  # 覆盖已存在的文件
        "-i", str(video_path),  # 输入文件
        "-ss", str(start_time),  # 开始时间
        "-t", str(duration),  # 持续时间
        "-c", "copy",  # 复制编码器，不重新编码
        str(output_path)  # 输出文件
    ]
    
    # 执行命令
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error extracting video segment: {e}")
        print(f"Command: {' '.join(cmd)}")
        print(f"Error output: {e.stderr.decode()}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--task_id",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--tgt_path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--debug",
        action="store_true",
    )
    args = parser.parse_args()

    task_id = args.task_id
    json_file = f"{args.src_path}/task_info/task_{args.task_id}.json"
    
    assert Path(json_file).exists, f"Cannot find {json_file}."
    main(args.src_path, args.tgt_path, task_id, json_file, args.debug)
