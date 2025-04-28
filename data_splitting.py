import os
import random
import shutil


def copy_files(files, src_images_dir, src_masks_dir, dst_images_dir, dst_masks_dir):
    for file in files:
        shutil.copy(os.path.join(src_images_dir, file + '.jpg'), os.path.join(dst_images_dir, file + '.jpg'))
        shutil.copy(os.path.join(src_masks_dir, file + '.png'), os.path.join(dst_masks_dir, file + '.png'))

def data_partion():
    
    images_dir = 'train/images'
    masks_dir = 'train/masks'
    train_images_dir = 'dataset/train/images'
    train_masks_dir = 'dataset/train/masks'
    val_images_dir = 'dataset/val/images'
    val_masks_dir = 'dataset/val/masks'


    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_masks_dir, exist_ok=True)


    image_files = [os.path.splitext(f)[0] for f in os.listdir(images_dir)]
    random.shuffle(image_files)


    split_ratio = 0.8
    split_index = int(len(image_files) * split_ratio)
    train_files = image_files[:split_index]
    val_files = image_files[split_index:]


    copy_files(train_files, images_dir, masks_dir, train_images_dir, train_masks_dir)
    copy_files(val_files, images_dir, masks_dir, val_images_dir, val_masks_dir)

    print(f"train: {len(train_files)} pictures")
    print(f"val: {len(val_files)} pictures")

data_partion()