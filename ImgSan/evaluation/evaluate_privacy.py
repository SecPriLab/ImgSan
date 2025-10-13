import os
import csv
import json
import math
import datetime
import numpy as np

from tqdm import tqdm
from pathlib import Path
from PIL import Image, ImageOps
from pycocotools import mask as maskUtils
from skimage.metrics import structural_similarity as ssim

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "PrivacyVQA2K"
IMAGES_DIR = DATA_DIR / "images"
ANNOTATIONS_FILE = DATA_DIR / "annotations.json"
OUTPUT_DIR = PROJECT_ROOT / "ImgSan" / "sanitized_images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OBF_INT_DIR = PROJECT_ROOT / "ImgSan" / "evaluation" / "obfuscation_intensities"
OBF_INT_DIR.mkdir(parents=True, exist_ok=True)

def load_mask_from_rle(rle):
    rle_decoded = {
            "size": rle["size"],
            "counts": rle["counts"].encode("utf-8")
        }
    mask = maskUtils.decode(rle_decoded)
    return mask

def get_bounding_box(mask):
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    return x_min, y_min, x_max, y_max

def apply_mask_black(arr, mask):
    arr = arr.copy()
    if arr.ndim == 3:
        arr[~mask, :] = 0
    else:
        arr[~mask] = 0
    return arr

def computed_masked_strength(npy_processed_path, mask):
    privacy_strength_array = np.load(npy_processed_path)
    mean_privacy_strength = privacy_strength_array[mask].mean()
    return mean_privacy_strength

def compute_masked_psnr(image1, image2, mask, max_pixel_value=255.0, eps=1e-10):
    image1 = image1.astype(np.float64)
    image2 = image2.astype(np.float64)

    mask_3c = np.repeat(mask[:, :, np.newaxis], 3, axis=2).astype(bool)

    diff_squared = (image1 - image2) ** 2

    masked_diff_squared = diff_squared[mask_3c]

    mse = masked_diff_squared.mean()

    if mse < eps:
        return float('inf')

    psnr = 10 * np.log10((max_pixel_value ** 2) / mse)
    return psnr

def evaluate_per_instance_privacy(original_path, processed_path, mask, image_name):
    image_original = Image.open(original_path)
    image_processed = Image.open(processed_path)
    image_original = ImageOps.exif_transpose(image_original)
    image_processed = ImageOps.exif_transpose(image_processed)
    arr_processed = np.array(image_processed)
    arr_processed_masked = apply_mask_black(arr_processed, mask)
    image_processed = Image.fromarray(arr_processed_masked)
    x_min, y_min, x_max, y_max = get_bounding_box(mask)
    image_gray_1 = image_processed.convert("L")
    image_gray_2 = image_original.convert("L")
    image_color_1 = image_processed.convert("RGB")
    image_color_2 = image_original.convert("RGB")
    image_gray_1 = np.array(image_gray_1)
    image_gray_2 = np.array(image_gray_2)
    image_color_1 = np.array(image_color_1)
    image_color_2 = np.array(image_color_2)
    image_gray_1_crop = image_gray_1[y_min:y_max+1, x_min:x_max+1]
    image_gray_2_crop = image_gray_2[y_min:y_max+1, x_min:x_max+1]
    image_color_1_crop = image_color_1[y_min:y_max+1, x_min:x_max+1, :]
    image_color_2_crop = image_color_2[y_min:y_max+1, x_min:x_max+1, :]
    mask_crop = mask[y_min:y_max+1, x_min:x_max+1]
    cur_psnr = compute_masked_psnr(image_color_1_crop, image_color_2_crop, mask_crop)
    cur_ssim_map = ssim(image_gray_1_crop, image_gray_2_crop, data_range=255, win_size = 21, full=True)[1]
    cur_ssim = (cur_ssim_map * mask_crop).mean()
    cur_privacy_strength = None
    npy_path = os.path.join(OBF_INT_DIR, f"{image_name.split('.', 1)[0]}.npy")
    if os.path.exists(npy_path):
        cur_privacy_strength = computed_masked_strength(npy_path, mask)

    return cur_psnr, cur_ssim, cur_privacy_strength

with open(ANNOTATIONS_FILE, "r", encoding="utf-8") as file:
    data = json.load(file)

all_psnrs = []
all_ssims = []
all_privacy_strengths = []

for item in tqdm(data, desc="Evaluating Privacy Metrics"):
    for privacy in item["rle_masks"]:
        mask = load_mask_from_rle(privacy)
        mask = mask.astype(bool)

        original_path = os.path.join(IMAGES_DIR, item["image"])
        processed_path = os.path.join(OUTPUT_DIR, item["image"])

        if not os.path.exists(processed_path):
            continue

        cur_psnr, cur_ssim, cur_privacy_strength = evaluate_per_instance_privacy(
            original_path, processed_path, mask, item["image"]
        )

        if cur_psnr is not None and not math.isinf(cur_psnr):
            all_psnrs.append(cur_psnr)
        if cur_ssim is not None:
            all_ssims.append(cur_ssim)
        if cur_privacy_strength is not None:
            all_privacy_strengths.append(cur_privacy_strength)

avg_psnr = np.mean(all_psnrs) if all_psnrs else 0
avg_ssim = np.mean(all_ssims) if all_ssims else 0
avg_privacy_strength = np.mean(all_privacy_strengths) if all_privacy_strengths else 0

average_metrics = {
    "average_psnr": avg_psnr,
    "average_ssim": avg_ssim,
    "average_privacy_strength": avg_privacy_strength
}

output_summary_path = PROJECT_ROOT / "ImgSan" / "evaluation" / "privacy_results.json"
output_summary_path.parent.mkdir(parents=True, exist_ok=True)

with open(output_summary_path, "w", encoding="utf-8") as f:
    json.dump(average_metrics, f, indent=4)


