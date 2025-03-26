import os
import shutil
from tqdm import tqdm

def custom_distribution_split(source_dir, 
                             target_dir,
                             num_classes=102,
                             first_batch=50):
    """
    自定义分配策略：
    1. 前N张图片放入第一个文件夹
    2. 剩余图片尽可能均匀分配到其他文件夹
    
    Args:
        source_dir: 原始图片目录
        target_dir: 目标目录
        num_classes: 总类别数（文件夹数量）
        first_batch: 第一个文件夹要放的图片数量
    """
    # 获取所有图片并排序
    image_files = sorted([f for f in os.listdir(source_dir)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)
    for class_id in range(num_classes):
        os.makedirs(os.path.join(target_dir, str(class_id)), exist_ok=True)

    # 第一阶段：填充第一个文件夹
    print(f"▶ 正在填充第一个文件夹（{first_batch}张）...")
    with tqdm(total=first_batch, desc="第一个文件夹") as pbar:
        for i in range(min(first_batch, len(image_files))):
            src = os.path.join(source_dir, image_files[i])
            dst = os.path.join(target_dir, "0", image_files[i])
            _safe_copy(src, dst)
            pbar.update(1)

    # 第二阶段：分配剩余图片到其他文件夹
    remaining_files = image_files[first_batch:]
    total_remaining = len(remaining_files)
    num_other_classes = num_classes - 1
    
    if num_other_classes > 0 and total_remaining > 0:
        print(f"▶ 正在分配剩余 {total_remaining} 张到 {num_other_classes} 个文件夹...")
        
        # 计算每个文件夹的基础数量和余数
        base_per_class = total_remaining // num_other_classes
        extra = total_remaining % num_other_classes
        
        file_idx = 0
        for class_id in tqdm(range(1, num_classes), desc="分配进度"):
            # 当前文件夹应分配的数量
            class_count = base_per_class + (1 if class_id <= extra else 0)
            
            # 执行复制
            for _ in range(class_count):
                if file_idx >= len(remaining_files):
                    break
                src = os.path.join(source_dir, remaining_files[file_idx])
                dst = os.path.join(target_dir, str(class_id), remaining_files[file_idx])
                _safe_copy(src, dst)
                file_idx += 1

def _safe_copy(src_path, dst_path):
    """处理文件名冲突的拷贝"""
    if not os.path.exists(src_path):
        return
    
    counter = 1
    while os.path.exists(dst_path):
        name, ext = os.path.splitext(os.path.basename(src_path))
        new_name = f"{name}_{counter}{ext}"
        dst_path = os.path.join(os.path.dirname(dst_path), new_name)
        counter += 1
    
    shutil.copy(src_path, dst_path)

if __name__ == '__main__':
    custom_distribution_split(
        source_dir=r'D:\tbw\Pytorch\Project\data\jpg',
        target_dir=r'D:\tbw\Pytorch\Project\data\custom_distribution',
        num_classes=102,
        first_batch=80  # 前50张放入第一个文件夹
    )