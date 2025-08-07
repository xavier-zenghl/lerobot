import argparse
import os
import shutil
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

def merge_lerobot_datasets(dataset_dirs, output_dir, chunk_size=1000, max_workers=8):
    output_dir = Path(output_dir)
    (output_dir / "data").mkdir(parents=True, exist_ok=True)
    (output_dir / "videos").mkdir(parents=True, exist_ok=True)
    (output_dir / "meta").mkdir(parents=True, exist_ok=True)

    merged_info = {
        "codebase_version": "v2.0",
        "robot_type": None,
        "total_episodes": 0,
        "total_frames": 0,
        "total_tasks": 0,
        "total_videos": 0,
        "total_chunks": 0,
        "chunks_size": chunk_size,
        "fps": 30,
        "splits": {},
        "features": None,
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
    }

    # --------- 1. 合并去重 tasks.jsonl，保留原始 task_index（按 task_index 升序）---------
    task_text_to_entry = {}
    for dataset_dir in dataset_dirs:
        tasks_path = Path(dataset_dir) / "meta/tasks.jsonl"
        with open(tasks_path, "r", encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line.strip())
                task_text = entry["task"]
                if task_text not in task_text_to_entry:
                    task_text_to_entry[task_text] = entry
    all_tasks = sorted(task_text_to_entry.values(), key=lambda x: x["task"])
    for i, task in enumerate(all_tasks):
        task["task_index"] = i

    # --------- 2. 收集所有 episode 的 parquet、视频路径和 meta ---------
    all_jobs = []
    views = [
        "images_dict.head.rgb", "images_dict.left.rgb", "images_dict.right.rgb",
        "images_dict.stereo_left.rgb", "images_dict.stereo_right.rgb"
    ]
    for dataset_dir in dataset_dirs:
        dataset_path = Path(dataset_dir)
        info_path = dataset_path / "meta/info.json"
        episodes_path = dataset_path / "meta/episodes.jsonl"

        # 记录 meta 信息
        with open(info_path, "r", encoding="utf-8") as f:
            info = json.load(f)
        if not merged_info["robot_type"]:
            merged_info["robot_type"] = info["robot_type"]
            merged_info["features"] = info["features"]
        merged_info["total_frames"] += info.get("total_frames", 0)
        merged_info["total_videos"] += info.get("total_videos", 0)

        # 所有 parquet，已排序
        parquet_files = sorted(dataset_path.glob("data/chunk-*/*.parquet"))
        # meta
        episodes_jsonl_entries = []
        if episodes_path.exists():
            with open(episodes_path, "r", encoding="utf-8") as f:
                for line in f:
                    ep = json.loads(line.strip())
                    episodes_jsonl_entries.append(ep)
        # 每个 episode 一一收集视频
        for i, parquet_file in enumerate(parquet_files):
            ep_meta = episodes_jsonl_entries[i] if i < len(episodes_jsonl_entries) else None
            video_dict = {}
            for view in views:
                # 视频和 parquet 一一对应，只要和 parquet 名称一致
                # videos/chunk-000/images_dict.head.rgb/episode_000000.mp4
                video_path = parquet_file.parent.parent / "videos" / view / f"{parquet_file.stem}.mp4"
                # 兼容部分数据结构，也可能直接在 dataset_path/videos/chunk-000/images_dict.head.rgb/episode_000000.mp4
                if not video_path.exists():
                    video_path = dataset_path / "videos" / parquet_file.parent.name / view / f"{parquet_file.stem}.mp4"
                video_dict[view] = video_path
            all_jobs.append((parquet_file, video_dict, ep_meta))

    # --------- 3. 合并并重命名拷贝 ---------
    def copy_episode_and_videos_with_index(args_indexed):
        i, (parquet_path, video_dict, ep_meta) = args_indexed
        episode_name = f"episode_{i:06d}"
        chunk_id = i // chunk_size
        chunk_dir = f"chunk-{chunk_id:03d}"

        # 拷贝 parquet
        new_parquet_dir = output_dir / "data" / chunk_dir
        new_parquet_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy(parquet_path, new_parquet_dir / f"{episode_name}.parquet")

        # 拷贝视频
        for view, video_src in video_dict.items():
            video_dst_dir = output_dir / "videos" / chunk_dir / view
            video_dst_dir.mkdir(parents=True, exist_ok=True)
            if video_src.exists():
                shutil.copy(video_src, video_dst_dir / f"{episode_name}.mp4")
            else:
                print(f"[WARN] 源视频不存在: {video_src}")

        # 构建新的 episodes.jsonl meta
        if ep_meta is not None:
            ep_meta_new = dict(ep_meta)  # 深拷贝
            ep_meta_new["episode_index"] = i
            ep_meta_new["episode_chunk"] = chunk_id
            return ep_meta_new
        else:
            return None

    # 多线程执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        new_episodes = list(tqdm(executor.map(copy_episode_and_videos_with_index, enumerate(all_jobs)), total=len(all_jobs)))

    # --------- 4. 合并 meta 结果写入 ---------
    episode_counter = len(all_jobs)
    merged_info["total_episodes"] = episode_counter
    merged_info["total_chunks"] = (episode_counter + chunk_size - 1) // chunk_size
    merged_info["total_tasks"] = len(all_tasks)
    merged_info["splits"] = {"train": f"0:{episode_counter}"}

    with open(output_dir / "meta/info.json", "w", encoding="utf-8") as f:
        json.dump(merged_info, f, indent=2)

    with open(output_dir / "meta/tasks.jsonl", "w", encoding="utf-8") as f:
        for task in all_tasks:
            f.write(json.dumps(task, ensure_ascii=False) + "\n")

    merged_episodes = [ep for ep in new_episodes if ep is not None]
    if merged_episodes:
        with open(output_dir / "meta/episodes.jsonl", "w", encoding="utf-8") as f:
            for ep in merged_episodes:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")

    print(f"\n✅ 合并完成，共 {episode_counter} 个 episode，输出路径: {output_dir}\n")

def fix_lerobot_index(dataset_dir):
    dataset_dir = Path(dataset_dir)
    tasks_path = dataset_dir / "meta/tasks.jsonl"
    tasks = []
    with open(tasks_path, 'r') as f:
        for line in f:
            tasks.append(json.loads(line))
            
    episode_list = sorted(list(dataset_dir.rglob('*.parquet')))
    index = 0
    for idx, data in enumerate(tqdm(episode_list)):
        df = pd.read_parquet(data)
        # fix task_index
        prompt = df['prompt'].to_numpy()[0]
        prompt = ''.join(chr(num) for num in prompt).split('\0')[0].strip()
        for task in tasks:
            if task['task'] == prompt:
                task_index = task['task_index']
                break
        df['task_index'] = pd.Series([task_index] * len(df['prompt']))
        
        # fix episode_index
        df['episode_index'] = pd.Series([idx] * len(df['prompt']))

        # fix index
        df['index'] = pd.Series(range(index, index + len(df['prompt'])))
        index += len(df['prompt'])
    
        df.to_parquet(data)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dirs', type=str, required=True, help='逗号分隔数据集')
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--chunk-size', type=int, default=1000)
    parser.add_argument('--max-workers', type=int, default=20)
    args = parser.parse_args()
    dirs = [d.strip() for d in args.dataset_dirs.split(',')]
    merge_lerobot_datasets(
        dataset_dirs=dirs,
        output_dir=args.output_dir,
        chunk_size=args.chunk_size,
        max_workers=args.max_workers
    )
    fix_lerobot_index(args.output_dir)