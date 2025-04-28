import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
from unet import UNet

COLOR_MAP = [
    [0, 0, 0],
    [204, 0, 0],
    [76, 153, 0],
    [204, 204, 0],
    [51, 51, 255],    # 4: l_eye
    [204, 0, 204],    # 5: r_eye
    [0, 255, 255],    # 6: l_brow
    [255, 204, 204],  # 7: r_brow
    [102, 51, 0],     # 8: l_ear
    [255, 0, 0],      # 9: r_ear
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

def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image

def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)

def postprocess_mask(mask):
    return mask.squeeze().cpu().numpy()

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

def save_aug(mask, save_path):
    mask_image = Image.fromarray(mask, mode='P')
    palette = [color for sublist in COLOR_MAP for color in sublist]
    mask_image.putpalette(palette)
    mask_image.save(save_path, format='PNG', optimize=True)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(19).to(device)
    model.load_state_dict(torch.load('./ckpt.pth', map_location=device)["state_dict"])
    model = model.to(device)
    model.eval()

    input_folder = './test/images'
    output_folder = './test/masks'

    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg'))]

    with torch.no_grad():
        for image_file in tqdm(image_files, desc="Predicting"):
            
            image_path = os.path.join(input_folder, image_file)
            image = load_image(image_path)
            input_tensor = preprocess_image(image).to(device)

            output = model(input_tensor)
            _, mask = torch.max(output, 1)

            mask = postprocess_mask(mask)

            save_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}.png")
            save_mask(mask, save_path)

    print(f"Predicted all {len(image_files)} images.")

if __name__ == "__main__":
    main()