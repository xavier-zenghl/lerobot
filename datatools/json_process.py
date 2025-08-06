import os
import sys
sys.path.append(os.getcwd())

import cv2
import h5py
import shutil
import random
import argparse
import numpy as np
import json

from tqdm import tqdm
from pathlib import Path
from typing import List, Union, Any
from multiprocessing import Process

from datatools.utils import merge_txt

# 需根据节点CPU算力情况进行调整，H100内部节点建议总进程数16个左右，本机建议4-8个节点
NUM_CPU = 1
PROC_PER_DEVICE = 1
target_size = (384, 384)

TARGET_DICT = {
    "open": "open the {}",
    "release": "release the {}",
    "grab": "pick up the {}",
    "put_it_in_the_plastic_bag": "place {} in the plastic bag",
    "back_home": "back to home{}",
}

OBJECT_DICT = {
}

def preprocess_object(obj):
    if obj in OBJECT_DICT:
        return OBJECT_DICT[obj]
    else:
        return obj.replace("_", " ")
        
def transform_annotation_to_instruction_old(data_list: List[Path], out_dir: str, data_dir: Path) -> None:
    for data in tqdm(data_list):
        with open(data, 'r') as f_r:
            annotations = json.load(f_r)
            for labels in annotations:
                src_file = labels['fileName'].replace('mp4', 'json')
                src_file = Path(out_dir) / src_file

                try:
                    clean_labels = []
                    for label in labels['result']['annotations'][0]['result']:
                        if 'attributes' not in label.keys():
                            clean_labels.append(
                                    {
                                        'start': label['start'],
                                        'end': label['end'],
                                        'prompt': 'None',
                                        'valid': False
                                    }
                                )
                            continue
                        if 'Abnormal trajectory' in label['attributes'].keys():
                            if label['attributes']['Abnormal trajectory']:
                                clean_labels.append(
                                    {
                                        'start': label['start'],
                                        'end': label['end'],
                                        'prompt': 'None',
                                        'valid': False
                                    }
                                )
                                continue
                        
                        subject = label['attributes']['subject'][0]
                        # 处理target的不同键值读取
                        target = label['attributes']['Target'][0]
                        if type(target) == dict:
                            target = target['name']
                        else:
                            target = target
                        # 处理process的不同键值读取
                        if 'process' in label['attributes'].keys():
                            process = label['attributes']['process']
                        else:
                            process = label['label']
                        # 处理object的不同键值读取
                        if 'objectobject' in label['attributes'].keys():
                            object = label['attributes']['objectobject'][0]
                        else:
                            object = label['attributes']['object'][0]
                            if type(object) == dict:
                                object = object['name']
                            else:
                                object = object
                                
                        if process == 'grab' or process == 'crawl':
                            prompt = 'the ' + subject + ' grab the ' + object
                        elif process == 'put down' or process == 'Let it go':
                            prompt = 'the ' + subject + ' put down the ' + object + ' to the ' + target
                        elif process == 'deliver' or process == 'switch hand' or process == 'Submit':
                            prompt = 'the ' + subject + ' deliver the ' + object + ' to the ' + target
                        else:
                            print(process)
                        clean_labels.append(
                            {
                                'start': label['start'],
                                'end': label['end'],
                                'prompt': prompt,
                                'valid': True
                            }
                        )
                except Exception as e:
                    print(src_file)
                with open(src_file, 'w') as f_w:
                    f_w.write(json.dumps(clean_labels, indent=4))
    
def transform_annotation_to_instruction(data_Path: Path) -> dict:
    with open(data_Path, 'r') as f_r:
        label_dict = {}
        annotations = json.load(f_r)
        for labels in annotations:
            frames = []
            # print(labels)
            src_file = labels['fileName'].replace('mp4', 'hdf5').replace('label_video_', '')
            src_file = data_Path.parent.parent / 'hdf5' / src_file

            try:
                for label in json.loads(labels['result'])['annotations'][0]['result']:
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
                    
                    ### 修改这一部分适应数据集
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
                            
                    ### 修改这一部分适应数据集

                    prompt = prompt.replace('_', ' ')
                    clean_labels = {
                            'frame': int(label['time'] * 30),
                            'prompt': prompt,
                            'valid': True
                    }
                    frames.append(clean_labels)
                
                label_dict[src_file] = frames
            except Exception as e:
                print(f'{src_file} error: {e}')
    
    return label_dict

                
def add_prompt_hdf5(data_list: List[Path]) -> None:
    labels = {}

    for data in tqdm(data_list):
        labels.update(transform_annotation_to_instruction(data))
   
    # with open(log_file, 'w') as f:
    for data, label in labels.items():
        prompts = []
        if not data.exists():
            print(f"{data} not exists")
            continue
        
        with h5py.File(data, 'r+') as src:
            length = src['time'].shape[0]
            if 'prompt' in src.keys():
                continue
                # del src['prompt']
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
        # f.write(f"{data}" + '\n')
        
        
def run(data_dir: str, out_dir: str) -> None:
    # if not os.path.exists(out_dir):
        # os.makedirs(out_dir, exist_ok=True)
        
    root_dir = Path(data_dir)
    data_list = list(root_dir.rglob('*).json'))
    random.shuffle(data_list)
    
    print(f"find json files: {len(data_list)}")
    
    add_prompt_hdf5(data_list)
    
    # num_proc = NUM_CPU * PROC_PER_DEVICE
    # num_per_proc = int(len(data_list) / num_proc) + 1
    
    # log_dir = Path(out_dir) / 'log'
    # log_dir.mkdir(exist_ok=True)
    # process_list = []
    # log_list = []
    
    # for i in range(NUM_CPU):
    #     for j in range(PROC_PER_DEVICE):
    #         process_id = i * PROC_PER_DEVICE + j
    #         start_idx = process_id * num_per_proc
    #         end_idx = start_idx + num_per_proc
    #         if end_idx > len(data_list):
    #             end_idx = len(data_list)

    #         slice_data = data_list[start_idx: end_idx]
    #         log_file = log_dir / f"process_{process_id}.txt"
    #         proc = Process(target=add_prompt_hdf5, args=(slice_data, out_dir, log_file))
    #         proc.start()
    #         process_list.append(proc)
    #         log_list.append(log_file)
    
    # for proc in process_list:
    #     proc.join()
            
    # result_txt = Path(out_dir) / 'result.txt'
    # merge_txt(log_list, result_txt)
    # print(f'process file list saved in {result_txt}')
    # shutil.rmtree(log_dir)

def arg_parse():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', dest='data_dir', default= '/home/extra/baifu/share_nfs/temp-data/Clean_table/', type=str, required=False, help='hdf5 file path to be loaded')
    argparser.add_argument('-o', dest='out_dir', default= 'dum_ori', type=str, required=False, help='time file path to be loaded')
    
    return argparser.parse_args()
                
if __name__ == '__main__':
    args = arg_parse()
    run(data_dir=args.data_dir, out_dir=args.out_dir)
