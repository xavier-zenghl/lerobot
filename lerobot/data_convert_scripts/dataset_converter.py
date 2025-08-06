import os
import sys
sys.path.append(os.getcwd())

import shutil
import traceback
import argparse
import logging
import time
import cv2
import json
import h5py
import torch
import numpy as np
# import pandas as pd
# import pyarrow as pa
# import tempfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed  # 新增

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from datatools.utils import extract_compress_file, xyz_quat_to_so3
from datatools.features_utils import DATASET_FEATURES

class S1HD5Extractor:
    @staticmethod
    def get_cameras(hdf5_data: h5py.File):
        rgb_cameras = []
        for key in hdf5_data["/images_dict/"]:
            if "rgb" in hdf5_data[f"/images_dict/{key}"]:
                rgb_cameras.append(key)
        return rgb_cameras

    @staticmethod
    def extract_episode_frames_split(
        episode_path: str | Path,
        features: dict[str, dict],
        hdf5_compressed: bool
    ) -> list[list[dict[str, torch.Tensor]]]:
        """
        读取单个 episode (HDF5 文件)，并返回一个「段落」列表；
        每个段落又是一个「帧」列表；每帧对应一个 dict。
        """
        groups = []
        with h5py.File(episode_path, "r") as file:
            # 基于 prompt 对帧做拆分
            print(file.keys())
            prompts = file['prompt'][:]
            last_index = 0
            cur_index = 0
            split_list = []
            while cur_index < len(prompts):
                prompt = prompts[cur_index].decode('utf-8')
                if prompt == 'None':
                    cur_index += 1
                    continue
                last_index = cur_index
                while cur_index < len(prompts) and prompts[cur_index].decode('utf-8') == prompt:
                    cur_index += 1
                split_list.append((last_index, cur_index))
                last_index = cur_index

            # 如果 hdf5_compressed=True，需要先把所有相机视频一起解压，避免多次重复解压
            total_mp4_frames = {}
            for feature_id in features:
                feature_name_hd5 = feature_id.replace(".", "/")
                if "rgb" in feature_name_hd5:
                    if hdf5_compressed:
                        mp4_frames = extract_compress_file(file[feature_name_hd5])
                    else:
                        # 如果并未采用 mp4 压缩，这里可以直接 numpy 读取
                        mp4_frames = np.array(file[feature_name_hd5])
                    total_mp4_frames[feature_name_hd5] = mp4_frames
                    
            try:        
                object = json.loads(file['object/bbox'][()].decode('utf-8'))
            except:
                traceback.print_exc()
                object = None
                
            # 按照分段提取数据
            for s in split_list:
                frames = []
                if s[1] - s[0] < 2:
                    continue
                for frame_idx in range(s[0], s[1]):
                    frame = {}
                    for feature_id in features:
                        feature_name_hd5 = feature_id.replace("/", ".").replace(".", "/")
                        if feature_name_hd5 not in file.keys():
                            continue
                        if "rgb" in feature_name_hd5:
                            image = total_mp4_frames[feature_name_hd5][frame_idx]
                            # HWC -> CHW，并转为 RGB 顺序
                            frame[feature_id] = image[..., ::-1]
                        elif 'prompt' in feature_id:
                            text = file[feature_name_hd5][frame_idx].decode('utf-8')
                            text_list = [ord(char) for char in text]
                            # 假设 prompt 长度固定 128
                            text_list += [32] * (128 - len(text_list))
                            frame[feature_id] = torch.from_numpy(np.array(text_list))
                        else:
                            frame[feature_id] = torch.from_numpy(file[feature_name_hd5][frame_idx])
                    frames.append(frame)
                groups.append(frames)
        return groups, object

    @staticmethod
    def extract_episode_frames_unsplit(
        episode_path: str | Path,
        features: dict[str, dict],
        hdf5_compressed: bool
    ) -> list[list[dict[str, torch.Tensor]]]:
        """
        读取单个 episode (HDF5 文件)，返回其对应帧列表，每一个文件都代表一个非切分的样本。
        """
        with h5py.File(episode_path, "r") as file:
            # 如果 hdf5_compressed=True，需要先把所有相机视频一起解压，避免多次重复解压
            total_mp4_frames = {}
            for feature_id in features:
                feature_name_hd5 = feature_id.replace(".", "/")
                if "rgb" in feature_name_hd5:
                    if hdf5_compressed:
                        mp4_frames = extract_compress_file(file[feature_name_hd5])
                    else:
                        # 如果并未采用 mp4 压缩，这里可以直接 numpy 读取
                        mp4_frames = np.array(file[feature_name_hd5])
                    total_mp4_frames[feature_name_hd5] = mp4_frames
                    
            try:        
                object = json.loads(file['object/bbox'][()].decode('utf-8'))
            except:
                # traceback.print_exc()
                object = None
                
            # 按照分段提取数据
            frames = []
            tasks_list = []
            for frame_idx in range(file["/joints_dict/joints_position_command"].shape[0]):
                frame = {}
                for feature_id in features:
                    feature_name_hd5 = feature_id.replace("/", ".").replace(".", "/")

                    if "rgb" in feature_name_hd5:
                        image = total_mp4_frames[feature_name_hd5][frame_idx]
                        # HWC -> CHW，并转为 RGB 顺序
                        frame[feature_id] = image[..., ::-1]
                    elif 'prompt' in feature_id:
                        text = file[feature_name_hd5][frame_idx].decode('utf-8')
                        text_list = [ord(char) for char in text]
                        # 假设 prompt 长度固定 128
                        text_list += [32] * (128 - len(text_list))
                        frame[feature_id] = torch.from_numpy(np.array(text_list))

                    elif feature_id == 'cartesian_so3_dict.cartesian_pose_command':
                        torso = xyz_quat_to_so3(file['command_poses_dict/astribot_torso'][frame_idx])
                        left_hand = xyz_quat_to_so3(file['command_poses_dict/astribot_arm_left'][frame_idx])
                        left_g = file['command_poses_dict/astribot_gripper_left'][frame_idx]
                        right_hand = xyz_quat_to_so3(file['command_poses_dict/astribot_arm_right'][frame_idx])
                        right_g = file['command_poses_dict/astribot_gripper_right'][frame_idx]
                        head = file['joints_dict/joints_position_command'][frame_idx][-2:]
                        chassis = file['joints_dict/joints_position_command'][frame_idx][:3]

                        merge = np.concatenate([torso, left_hand, left_g, right_hand, right_g, head, chassis])

                        frame[feature_id] = torch.from_numpy(merge)
                    
                    elif feature_id == 'cartesian_so3_dict.cartesian_pose_state':
                        torso = xyz_quat_to_so3(file['poses_dict/astribot_torso'][frame_idx])
                        left_hand = xyz_quat_to_so3(file['poses_dict/astribot_arm_left'][frame_idx])
                        left_g = file['poses_dict/astribot_gripper_left'][frame_idx]
                        right_hand = xyz_quat_to_so3(file['poses_dict/astribot_arm_right'][frame_idx])
                        right_g = file['poses_dict/astribot_gripper_right'][frame_idx]
                        head = file['joints_dict/joints_position_state'][frame_idx][-2:]
                        chassis = file['joints_dict/joints_position_state'][frame_idx][:3]

                        merge = np.concatenate([torso, left_hand, left_g, right_hand, right_g, head, chassis])
                        frame[feature_id] = torch.from_numpy(merge)
                    elif feature_name_hd5 not in file.keys():
                        continue
                    else:
                        frame[feature_id] = torch.from_numpy(file[feature_name_hd5][frame_idx])
                frames.append(frame)
                tasks_list.append([file['prompt'][frame_idx].decode('utf-8')])
        return frames, object, tasks_list

    @staticmethod
    def extract_episode_frames_general(
        episode_path: str | Path,
        features: dict[str, dict],
        hdf5_compressed: bool
    ):
        """
        读取单个 episode (HDF5 文件)，返回其对应帧列表，每一个文件都代表一个非切分的样本。
        """
        frames = []
        with h5py.File(episode_path, "r") as file:
            total_mp4_frames = {}
            for feature_id in features:
                feature_name_hd5 = feature_id.replace(".", "/")
                if "rgb" in feature_name_hd5:
                    if hdf5_compressed:
                        mp4_frames = extract_compress_file(file[feature_name_hd5])
                    else:
                        # 如果并未采用 mp4 压缩，这里可以直接 numpy 读取
                        mp4_frames = np.array(file[feature_name_hd5])
                    total_mp4_frames[feature_name_hd5] = mp4_frames
            if 'object' in file.keys():      
                description = file['object/description'][()][0].decode('utf-8')
                object = json.loads(file['object/bbox'][()].decode('utf-8'))
            else:
                description = 'None'
                object = None
            episode_idx = episode_path.stem.split("_")[-1]
            for frame_idx in range(file["/joints_dict/joints_position_command"].shape[0]):
                frame = {}
                for feature_id in features:
                    feature_name_hd5 = feature_id.replace(".", "/")
                    if "rgb" in feature_id.split("."):
                        if hdf5_compressed:
                            # 已经预先解压了 mp4_frames
                            image = total_mp4_frames[feature_name_hd5][frame_idx]
                        else:
                            image = file[feature_name_hd5][frame_idx]
                        frame[feature_id] = image[..., ::-1]
                    elif 'prompt' in feature_id:
                        # 旧版本的prompt
                        # text_matched = prompt_list[prompt_list["idx"]==int(episode_idx)]
                        # text = text_matched['instruction'].values[0] + '\0'

                        # 新版本的prompt
                        if feature_name_hd5 in file.keys():
                            text = file[feature_name_hd5][frame_idx].decode('utf-8')
                        else:
                            text = description
                        # if text == 'None':
                        #     text = 'return to the initial position'
                        
                        # prompt转换为ascii码
                        text_list = [ord(char) for char in text]
                        text_list = text_list + [32]*(128-len(text_list))
                        frame[feature_id] = np.array(text_list)
                    elif 'cartesian' in feature_id:
                        continue
                    else:
                        frame[feature_id] = np.array(file[feature_name_hd5][frame_idx])

                frames.append(frame)
        return frames, (description, object)
    
    def extract_episode_frames_generator(
        episode_path: str | Path,
        features: dict[str, dict],
        hdf5_compressed: bool
    ):
        """
        生成器版本，保持HDF5文件打开状态
        """
        file = h5py.File(episode_path, "r")
        try:
            # 获取分段信息
            prompts = file['prompt'][:]
            last_index = 0
            cur_index = 0
            split_list = []
            while cur_index < len(prompts):
                prompt = prompts[cur_index].decode('utf-8')
                if prompt == 'None':
                    cur_index += 1
                    continue
                last_index = cur_index
                while cur_index < len(prompts) and prompts[cur_index].decode('utf-8') == prompt:
                    cur_index += 1
                split_list.append((last_index, cur_index))
                last_index = cur_index

            # 如果 hdf5_compressed=True，需要先把所有相机视频一起解压，避免多次重复解压
            total_mp4_frames = {}
            for feature_id in features:
                feature_name_hd5 = feature_id.replace(".", "/")
                if "rgb" in feature_name_hd5:
                    if hdf5_compressed:
                        mp4_frames = extract_compress_file(file[feature_name_hd5])
                    else:
                        # 如果并未采用 mp4 压缩，这里可以直接 numpy 读取
                        mp4_frames = np.array(file[feature_name_hd5])
                    total_mp4_frames[feature_name_hd5] = mp4_frames
                    
            # 按照分段提取数据
            for s in split_list:
                frames = []
                if s[1] - s[0] < 2:
                    continue
                for frame_idx in range(s[0], s[1]):
                    frame = {}
                    for feature_id in features:
                        feature_name_hd5 = feature_id.replace("/", ".").replace(".", "/")
                        if feature_name_hd5 not in file.keys():
                            continue
                        if "rgb" in feature_name_hd5:
                            # 已经预先解压了 mp4_frames
                            image = total_mp4_frames[feature_name_hd5][frame_idx]
                            frame[feature_id] = image[..., ::-1]
                        elif 'prompt' in feature_id:
                            text = file[feature_name_hd5][frame_idx].decode('utf-8')
                            text_list = [ord(char) for char in text]
                            text_list += [32] * (128 - len(text_list))
                            frame[feature_id] = torch.from_numpy(np.array(text_list))
                        else:
                            frame[feature_id] = torch.from_numpy(file[feature_name_hd5][frame_idx])
                    frames.append(frame)
                yield frames
        finally:
            file.close()

    @staticmethod
    def define_features(hdf5_file_path: Path, hdf5_compressed: bool, encode_as_video: bool = True):
        """
        从某个HDF5文件中抽取数据维度信息，用于初始化 LeRobotDataset。
        """
                # Initialize lists to store topics and features
        topics = []
        features = {}
        feature_dict = DATASET_FEATURES['astribot']

        # Open the HDF5 file
        with h5py.File(hdf5_file_path, "r") as hdf5_file:
            # Collect all dataset names in the HDF5 file
            hdf5_file.visititems(lambda name, obj: topics.append(name) if isinstance(obj, h5py.Dataset) else None)
            
            # Iterate over each topic to define its features
            for topic in topics:
                # If the topic is an image, define it as a video feature
                if "rgb" in topic.split("/"):

                    sample = hdf5_file[topic]
                    if "height" not in sample.attrs.keys() or "width" not in sample.attrs.keys():
                        continue
                    height = sample.attrs.get("height")
                    width = sample.attrs.get("width")
                    features[topic.replace("/", ".")] = {
                        "dtype": "video" if encode_as_video else "image",
                        "shape": (3, int(height), int(width)),
                        "names": [
                            "channel",
                            "height",
                            "width",
                        ],
                    }

                # Skip compressed length topics
                elif "compress_len" in topic.split("/"):
                    continue
                # Otherwise, define it as a regular feature
                elif 'timestamp' in topic:
                    continue
                elif "joints_dict" in topic.split("/"):
                    if topic.replace("/", ".") not in feature_dict:
                        continue
                    features[topic.replace("/", ".")] = {
                        "dtype": str(hdf5_file[topic][0].dtype),
                        "shape": (topic_shape := hdf5_file[topic][0].shape),
                        "names": feature_dict[topic.replace("/", ".")],
                    }

                elif topic == "poses_dict/merge_pose":
                    features["cartesian_so3_dict.cartesian_pose_state"] = {
                        "dtype": str(hdf5_file[topic][0].dtype),
                        "shape": (34, ),
                        "names": feature_dict["cartesian_so3_dict.cartesian_pose_state"],
                    }

                elif topic == "command_poses_dict/merge_pose":
                    features["cartesian_so3_dict.cartesian_pose_command"] = {
                        "dtype": str(hdf5_file[topic][0].dtype),
                        "shape": (34,),
                        "names": feature_dict["cartesian_so3_dict.cartesian_pose_command"],
                    }

                else:
                    continue
        return features

class DatasetConverter:
    def __init__(
        self,
        raw_path: Path | str,
        root_path: Path | str,
        dataset_repo_id: str,
        fps: int,
        epoch_num: int = -1,
        robot_type: str = "",
        encode_as_videos: bool = True,
        image_writer_processes: int = 0,
        image_writer_threads: int = 0,
        hdf5_compressed: bool = True,
        split: bool = True,
    ):
        self.raw_path = raw_path if isinstance(raw_path, Path) else Path(raw_path)
        self.root_path = root_path if isinstance(root_path, Path) else Path(root_path)
        self.dataset_repo_id = dataset_repo_id
        self.fps = fps
        self.robot_type = robot_type
        self.image_writer_threads = image_writer_threads
        self.image_writer_processes = image_writer_processes
        self.encode_as_videos = encode_as_videos
        self.hdf5_compressed = hdf5_compressed
        self.split = split

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - [%(name)s] - %(message)s")
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.info(f"{'-'*10} Aloha HD5 -> Lerobot Converter {'-'*10}")
        self.logger.info(f"Processing from {self.raw_path}")
        self.logger.info(f"Dataset: {self.dataset_repo_id}, FPS: {self.fps}, Robot: {self.robot_type}")
        self.logger.info(f"encode_as_videos: {self.encode_as_videos}")
        self.logger.info(f"Processes: {self.image_writer_processes}, Threads: {self.image_writer_threads}")
        print(self.raw_path)
        self.episode_list = list(self.raw_path.rglob("*.hdf5"))
        print(f"Found {len(self.episode_list)} episodes.")
        self.epoch_num = len(self.episode_list)
        
        # 只从第一集获取特征信息
        self.features = S1HD5Extractor.define_features(
            self.episode_list[0],
            encode_as_video=self.encode_as_videos,
            hdf5_compressed=self.hdf5_compressed
        )

    def init_lerobot_dataset(self, resume: bool = False):
        # from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        if os.path.exists(self.root_path / self.dataset_repo_id):
            if resume:
                self.dataset = LeRobotDataset(
                    root=self.root_path / self.dataset_repo_id,
                    repo_id=self.dataset_repo_id,
                    local_files_only=True
                )
            else:
                shutil.rmtree(self.root_path / self.dataset_repo_id)
        self.dataset = LeRobotDataset.create(
            root=self.root_path / self.dataset_repo_id,
            repo_id=self.dataset_repo_id,
            fps=self.fps,
            robot_type=self.robot_type,
            features=self.features,
            image_writer_threads=self.image_writer_threads,
            image_writer_processes=self.image_writer_processes,
        )
        return self.dataset

    def extract_episodes(self, episode_description: str = ""):
        if self.split:
            self.extract_episodes_split(episode_description)
        else:
            self.extract_episodes_unsplit(episode_description)
    
    def extract_episodes_split(self, episode_description: str = ""):
        for idx, episode_path in enumerate(self.episode_list):
            try:
                self.logger.info(f"Processing episode: {idx+1}/{self.epoch_num} => {episode_path.name}")
                groups, object = S1HD5Extractor.extract_episode_frames_split(
                    episode_path, self.features, self.hdf5_compressed
                )
                for group in groups:
                    for frame in group:
                        self.dataset.add_frame_no_save(frame)
                    if frame.get("prompt") is not None:
                        task_description = frame.get("prompt")
                        # 解码过程
                        string = ''.join(chr(num) for num in task_description)
                        # 去除结束符号
                        task_description = string.split('\0')[0].strip()  # 截取结束符号前的部分
                    self.logger.info(f"Description: {task_description} ...")
                    self.dataset.save_episode_no_save(task=task_description, object=object)
            except Exception as e:
                self.logger.error(f"[Error] {episode_path} => {e}")
                traceback.print_exc()
        self.dataset.consolidate(run_compute_stats=False)
    
    def extract_episodes_unsplit(self, episode_description: str = ""):
        for idx, episode_path in enumerate(self.episode_list):
            try:
                self.logger.info(f"Processing episode: {idx+1}/{self.epoch_num} => {episode_path.name}")
                frames, object, tasks_list = S1HD5Extractor.extract_episode_frames_unsplit(
                    episode_path, self.features, self.hdf5_compressed
                )
                
                for frame, task in zip(frames, tasks_list):
                    if task[0] == 'None':
                        continue
                    self.dataset.add_frame_no_save(frame, task, coarse_task=episode_description)

                self.logger.info(f"Description: {episode_description} ...")
                self.dataset.save_episode_no_save(task=episode_description, object=object)
            except Exception as e:
                self.logger.error(f"[Error] {episode_path} => {e}")
                traceback.print_exc()
        self.dataset.consolidate(run_compute_stats=False)

    def extract_episodes_general(self, task_description: str = ""):
        """
        原有串行版本
        """
        for idx, episode_path in enumerate(self.episode_list):
            try:
                self.logger.info(f"Processing episode: {idx+1}/{self.epoch_num} => {episode_path.name}")
                group, object = S1HD5Extractor.extract_episode_frames_general(
                    episode_path, self.features, self.hdf5_compressed
                )
                for frame in group:
                    self.dataset.add_frame_no_save(frame)

                episode_idx = episode_path.stem.split("_")[-1]
                description, bbox = object
                self.logger.info(f"Saving Episode:{episode_idx} with Description: {description} ...")
                self.dataset.save_episode_no_save(task=description, object=bbox)
            except Exception as e:
                self.logger.error(f"[Error] {episode_path} => {e}")
                traceback.print_exc()
        self.dataset.consolidate(run_compute_stats=False)
