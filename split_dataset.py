import os
import shutil
import random

def split_dataset(images_dir, labels_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    # 创建目标目录
    os.makedirs('block_data/images/train', exist_ok=True)
    os.makedirs('block_data/images/val', exist_ok=True)
    os.makedirs('block_data/images/test', exist_ok=True)
    os.makedirs('block_data/labels/train', exist_ok=True)
    os.makedirs('block_data/labels/val', exist_ok=True)
    os.makedirs('block_data/labels/test', exist_ok=True)

    # 获取所有图像文件
    images = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    random.shuffle(images)
    
    # 计算分割点
    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    # 分割
    for i, img in enumerate(images):
        label = img.replace('.jpg', '.txt')
        if i < train_end:
            split = 'train'
        elif i < val_end:
            split = 'val'
        else:
            split = 'test'
        shutil.copy(os.path.join(images_dir, img), f'block_data/images/{split}/{img}')
        shutil.copy(os.path.join(labels_dir, label), f'block_data/labels/{split}/{label}')

    print("Dataset split completed!")

if __name__ == "__main__":
    split_dataset('block_data/images', 'block_data/labels')