import os
import torch
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import random
import numpy as np
from torchvision.utils import make_grid
from PIL import Image
from predict import postprocess_mask, save_aug



class CustomTransform:
    def __init__(self, is_train=True):
        self.is_train = is_train
        self.left_eye_color = 4
        self.right_eye_color = 5
        self.left_brow_color = 6
        self.right_brow_color = 7
        self.left_ear_color = 8
        self.right_ear_color = 9

    def __call__(self, image, mask):
        if self.is_train:
            image = TF.hflip(image)
            mask = self.flip_mask(mask)

            angle = random.uniform(-10, 10)

            image = TF.rotate(image, angle)
            if isinstance(mask, np.ndarray):
                mask = Image.fromarray(mask)
            mask = TF.rotate(mask, angle, fill=(0))
            mask = np.array(mask)

            image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))
            image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))
            image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.8, 1.2))
            image = TF.adjust_hue(image, hue_factor=random.uniform(-0.1, 0.1))

        return image, mask

    def flip_mask(self, mask):
        mask_np = np.array(mask)
        flipped_mask = np.fliplr(mask_np)

        # Swap the left and right colors
        left_eye_mask = flipped_mask == self.left_eye_color
        right_eye_mask = flipped_mask == self.right_eye_color
        left_brow_mask = flipped_mask == self.left_brow_color
        right_brow_mask = flipped_mask == self.right_brow_color
        left_ear_mask = flipped_mask == self.left_ear_color
        right_ear_mask = flipped_mask == self.right_ear_color

        flipped_mask[left_eye_mask] = self.right_eye_color
        flipped_mask[right_eye_mask] = self.left_eye_color
        flipped_mask[left_brow_mask] = self.right_brow_color
        flipped_mask[right_brow_mask] = self.left_brow_color
        flipped_mask[left_ear_mask] = self.right_ear_color
        flipped_mask[right_ear_mask] = self.left_ear_color

        return flipped_mask.astype(np.uint8)


def image_masks_augmentation(image_path, mask_path, num_augmentations=5):
    
    original_image = Image.open(image_path).convert('RGB')
    original_mask = Image.open(mask_path)

    transform = CustomTransform(is_train=True)

    flip_image, flip_mask = transform(original_image, original_mask)

    return flip_image, flip_mask



images_path = 'train/images'
masks_path = 'train/masks'

if not os.path.exists(images_path):
    os.makedirs(images_path)
if not os.path.exists(masks_path):
    os.makedirs(masks_path)

i = 0
for file_name in os.listdir(images_path):
    original_image_path = os.path.join(images_path, file_name)

    mask_name = os.path.splitext(file_name)[0] + ".png"
    original_mask_path = os.path.join(masks_path, mask_name)

    flip_image, flip_mask = image_masks_augmentation(original_image_path, original_mask_path)

    aug_image_path = os.path.join(images_path, f'{i}.jpg')
    aug_mask_path = os.path.join(masks_path, f'{i}.png')
    i += 1

    flip_image.save(aug_image_path)
    save_aug(flip_mask, aug_mask_path)
