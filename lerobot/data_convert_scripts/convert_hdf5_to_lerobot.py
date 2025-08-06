import os
import sys
sys.path.append(os.getcwd())

import argparse
import time
from pathlib import Path

from lerobot.data_convert_scripts.dataset_converter import DatasetConverter

import random
import argparse

from tqdm import tqdm
from pathlib import Path

from multiprocessing import Process


def main():
    parser = argparse.ArgumentParser(description="Convert Aloha HD5 dataset and push to Hugging Face hub.")
    parser.add_argument("--raw-path", type=Path, required=True,
                        help="Directory containing the raw hdf5 files.")
    parser.add_argument("--root-path", type=Path, required=True, 
                        help="Root path for the dataset.")
    parser.add_argument("--epoch_num", type=int, required=False, default=-1, help="Number of episodes to process.")
    parser.add_argument("--dataset-repo-id", type=str, required=True, 
                        help="Repository ID where the dataset will be stored.")
    parser.add_argument("--fps", type=int, required=False, default=30,
                        help="Frames per second for the dataset.")
    parser.add_argument("--hdf5_compressed", type=lambda v: v.lower() in ("true", "1"),
                        required=False, default="true", help="Set to True if the hdf5 are compressed.")
    parser.add_argument("--split", type=lambda v: v.lower() in ("true", "1"),
                        required=False, default="true", help="Set to True if the hdf5 are split.")
    parser.add_argument("--description", type=str, help="Description of the dataset.",
                        default="S1 pick the obj and place it into the box.")
    parser.add_argument("--robot-type", type=str, choices=["S1-stationary", "S1-mobile"],
                        default="S1-stationary", help="Type of robot.")
    parser.add_argument("--private", type=lambda v: v.lower() in ("true", "1"), default=True,
                        help="Set to True to make the dataset private.")
    parser.add_argument("--push", type=lambda v: v.lower() in ("true", "1"), default=False,
                        help="Set to True to push videos to the hub.")
    parser.add_argument("--license", type=str, default="apache-2.0", help="License for the dataset.")
    parser.add_argument("--image-compressed", type=lambda v: v.lower() in ("true", "1"), default=False,
                        help="Set to True if the images are compressed.")
    parser.add_argument("--video-encoding", type=lambda v: v.lower() in ("true", "1"), default=True,
                        help="Set to True to encode images as videos.")
    parser.add_argument("--nproc", type=int, default=2, help="Number of image writer processes.")
    parser.add_argument("--nthreads", type=int, default=2, help="Number of image writer threads.")
    parser.add_argument("--resume", action='store_true', help='Resume from existing dataset directory.')
    args = parser.parse_args()
    print(f'Dataset: {args.raw_path}...')
    # time.sleep(2)

    converter = DatasetConverter(
        raw_path=args.raw_path,
        root_path=args.root_path,
        epoch_num=args.epoch_num,
        dataset_repo_id=args.dataset_repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        encode_as_videos=args.video_encoding,
        image_writer_processes=args.nproc,
        image_writer_threads=args.nthreads,
        hdf5_compressed=args.hdf5_compressed,
        split=args.split
    )
    converter.init_lerobot_dataset(args.resume)

    t0 = time.time()
    # 串行处理版本
    converter.extract_episodes(episode_description=args.description)
    print(f"Total time cost: {time.time() - t0:.2f} sec.")


if __name__ == "__main__":
    main()
