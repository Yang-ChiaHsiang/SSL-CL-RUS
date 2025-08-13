import os
import numpy as np
import cv2
from PIL import Image, ImageOps
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(description='Training parameters')
    parser.add_argument('--dataset_path', type=str, default='~/Dataset/0_data_dataset_voc_950', help='Path to the dataset')
    parser.add_argument('--voc_output_dir', type=str, default='dataset/splits/kidney', help='Output directory for results')
    parser.add_argument('--voc_splits', type=str, default='900', help='splits')
    parser.add_argument('--crop_output_dir', type=str, default='data/0_data_dataset_voc_950', help='crop_output_dir')
    parser.add_argument('--img_size', type=int, default=224, help='Size of the input images')
    return parser.parse_args()

def _enhanced_find_and_crop(image_pil, threshold=30, min_area_ratio=0.3, width_ratio_thresh=0.9):
    image = np.array(image_pil.convert("RGB"))
    orig_height, orig_width = image.shape[:2]

    # CLAHE 增強對比
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray_normalized = clahe.apply(gray)

    # 二值化（低、高閾值 OR）
    _, binary_low = cv2.threshold(gray_normalized, 20, 255, cv2.THRESH_BINARY)
    _, binary_high = cv2.threshold(gray_normalized, 60, 255, cv2.THRESH_BINARY)
    binary = cv2.bitwise_or(binary_high, binary_low)

    # 形態學操作
    kernel = np.ones((5,5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("No contours found.")
        return image_pil, None

    max_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(max_contour)
    area = w * h
    width_ratio = w / orig_width

    if area < min_area_ratio * (orig_height * orig_width):
        print("區域太小，使用原圖")
        region = image
        x, y, w, h = 0, 0, orig_width, orig_height
    elif width_ratio > width_ratio_thresh:
        print("寬度接近原圖，裁剪上方15%和右方15%")
        crop_height = int(orig_height * 0.85)
        crop_width = int(orig_width * 0.80)
        region = image[int(orig_height * 0.10):orig_height, 0:crop_width]
        x, y, w, h = 0, int(orig_height * 0.10), crop_width, crop_height
    else:
        region = image[y:y+h, x:x+w]

    max_dim = max(region.shape[0], region.shape[1])
    square = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    y_offset = (max_dim - region.shape[0]) // 2
    x_offset = (max_dim - region.shape[1]) // 2
    square[y_offset:y_offset+region.shape[0], x_offset:x_offset+region.shape[1]] = region

    square_pil = Image.fromarray(square)
    return square_pil, (x, y, w, h)

def _cropImage(image, x, y, w, h):
    return image.crop((x, y, x + w, y + h))

def preprocessing(id_path, dataset_path, crop_output_dir, img_size):
    with open(id_path, 'r') as file:
        lines = file.readlines()
        
    for line in lines:
        img_path, mask_path = line.strip().split()
        og_img_path = os.path.join(dataset_path, img_path)
        mask_img_path = os.path.join(dataset_path, mask_path)
        
        try:
            image = Image.open(og_img_path)
            mask = Image.open(mask_img_path)
            
            cropped_image, contour = _enhanced_find_and_crop(image)
            if contour is not None:
                x, y, w, h = contour
                cropped_mask = _cropImage(mask, x, y, w, h)
                cropped_mask = cropped_mask.resize((img_size, img_size), Image.NEAREST)
            else:
                cropped_mask = mask.resize((img_size, img_size), Image.NEAREST)

            cropped_image = cropped_image.resize((img_size, img_size), Image.BILINEAR)
            
            save_image_path = os.path.join(crop_output_dir, img_path)
            save_mask_path = os.path.join(crop_output_dir, mask_path)
            
            save_image_dir = os.path.dirname(save_image_path)
            save_mask_dir = os.path.dirname(save_mask_path)
            if not os.path.exists(save_image_dir):
                os.makedirs(save_image_dir)
            if not os.path.exists(save_mask_dir):
                os.makedirs(save_mask_dir)
                
            cropped_image.save(save_image_path)
            cropped_mask.save(save_mask_path)
            print(f'Cropped image saved to {save_image_path}')
            print(f'Cropped mask saved to {save_mask_path}')
        except Exception as e:
            print(f'Error processing {img_path}: {e}')

def main():
    args = parse_args()
    dataset_path = args.dataset_path
    voc_output_dir = args.voc_output_dir
    voc_splits = args.voc_splits
    crop_output_dir = args.crop_output_dir
    img_size = args.img_size
    
    if not os.path.exists(crop_output_dir):
        os.makedirs(crop_output_dir)
        
    train_id_path = f'{voc_output_dir}/train.txt'
    val_id_path = f'{voc_output_dir}/val.txt'
    test_id_path = f'{voc_output_dir}/test.txt'
    unlabel_id_path = f'{voc_output_dir}/{voc_splits}/unlabeled.txt'
    
    preprocessing(train_id_path, dataset_path, crop_output_dir, img_size)
    preprocessing(val_id_path, dataset_path, crop_output_dir, img_size)
    preprocessing(test_id_path, dataset_path, crop_output_dir, img_size)
    preprocessing(unlabel_id_path, dataset_path, crop_output_dir, img_size)
    
if __name__ == '__main__':
    main()