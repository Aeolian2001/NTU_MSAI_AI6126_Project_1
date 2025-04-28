# -*- coding: utf-8 -*-
#
# @File:   run.py
# @Author: Haozhe Xie
# @Date:   2025-02-18 19:09:59
# @Last Modified by: Haozhe Xie
# @Last Modified at: 2025-02-19 08:13:52
# @Email:  root@haozhexie.com

import argparse
import cv2
import torch
from unet import UNet
from PIL import Image
from torchvision import transforms
import numpy as np

COLOR_MAP = [
    [0, 0, 0],
    [204, 0, 0],
    [76, 153, 0],
    [204, 204, 0],
    [51, 51, 255],  # 4: l_eye
    [204, 0, 204],  # 5: r_eye
    [0, 255, 255],  # 6: l_brow
    [255, 204, 204],  # 7: r_brow
    [102, 51, 0],  # 8: l_ear
    [255, 0, 0],  # 9: r_ear
    [102, 204, 0],
    [255, 255, 0],
    [0, 0, 153],
    [0, 0, 204],
    [255, 51, 153],
    [0, 204, 204],
    [0, 51, 0],
    [255, 153, 51],
    [0, 204, 0],
]


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)


def apply_color_map(mask):
    color_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.uint8)
    for class_idx, _ in enumerate(COLOR_MAP):
        color_mask[mask == class_idx] = class_idx
    return color_mask


def save_mask(mask, save_path):
    colored_mask = apply_color_map(mask)
    mask_image = Image.fromarray(colored_mask, mode='P')
    palette = [color for sublist in COLOR_MAP for color in sublist]
    mask_image.putpalette(palette)
    mask_image.save(save_path, format='PNG', optimize=True)


def main(input, output, weights):
    # Load the input image
    img = Image.open(input).convert('RGB')
    # img = cv2.imread(input)

    # TODO: Initialize the neural network model
    # Example:
    # from models import YourSegModel
    # model = YourSegModel()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(19).to(device)
    # Load the checkpoint
    ckpt = torch.load(weights)
    # NOTE: Make sure that the weights are saved in the "state_dict" key
    # DO NOT CHANGE THIS VALUE, i.e., ckpt["state_dict"]
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device)
    model.eval()

    # Inference with the model (Update as needed)
    # Normalize the image.
    # NOTE: Make sure it is aligned with the training data
    # Example: img = (img / 255.0 - 0.5) * 2.0
    input_tensor = preprocess_image(img).to(device)
    prediction = model(input_tensor)

    # Convert PyTorch Tensor to numpy array
    # mask = prediction.cpu().numpy()

    _, mask = torch.max(prediction, 1)
    mask = mask.squeeze().cpu().numpy()

    # Save the prediction
    save_mask(mask, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    parser.add_argument("--weights", type=str, default="ckpt.pth")
    args = parser.parse_args()
    main(args.input, args.output, args.weights)
