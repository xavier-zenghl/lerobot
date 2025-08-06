import os
import shutil
from pathlib import Path

def split_folder(hdf5_dir, n_split=8, data_format='hdf5'):
    hdf5_dir = Path(hdf5_dir)
    # 删除现有的所有 split_* 文件夹
    for item in hdf5_dir.iterdir():
        if item.is_dir() and item.name.startswith('split_'):
            shutil.rmtree(item)

    # 获取所有子文件夹（非文件）
    if data_format == 'directory':
        files = sorted([f for f in hdf5_dir.iterdir() if f.is_dir()])
    else:
        files = sorted([f for f in hdf5_dir.glob(f"*.{data_format}")])
        
    subfolders = []
    split_files = [[] for _ in range(n_split)]
    
    for idx, f in enumerate(files):
        split_files[idx % n_split].append(f)
    for i in range(n_split):
        sub_dir = hdf5_dir / f"split_{i+1}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        for f in split_files[i]:
            link = sub_dir / f.name
            if not link.exists():
                # os.symlink(f.resolve(), link)
                shutil.copy(f, link)
        if split_files[i]:  # 只打印非空目录
            subfolders.append(str(sub_dir))

if __name__ == "__main__":
    import sys
    hdf5_dir = sys.argv[1]
    n_split = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    data_format = sys.argv[3] if len(sys.argv) > 3 else 'hdf5'
    split_folder(hdf5_dir, n_split, data_format)
