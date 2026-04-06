import os
import cv2
import numpy as np
import random

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
INPUT_PATH  = "Dataset_Preprocessed"   
OUTPUT_PATH = "Dataset_Augmented"      
TARGET_PER_CLASS = 100                 
RANDOM_STATE = 42
random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────
# FUNGSI AUGMENTASI
# ─────────────────────────────────────────────

def flip_horizontal(img):
    """Cerminkan gambar secara horizontal."""
    return cv2.flip(img, 1)


def rotate(img, angle_range=(-15, 15)):
    """Rotasi gambar dengan sudut acak dalam rentang tertentu."""
    angle = random.uniform(*angle_range)
    h, w  = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                              borderMode=cv2.BORDER_REFLECT)
    return rotated


def adjust_brightness(img, factor_range=(0.6, 1.4)):
    """Ubah kecerahan gambar secara acak."""
    factor = random.uniform(*factor_range)
    result = np.clip(img.astype(np.float32) * factor, 0, 255)
    return result.astype(np.uint8)


def add_gaussian_noise(img, std_range=(5, 20)):
    """Tambahkan noise Gaussian untuk simulasi kondisi kamera."""
    std   = random.uniform(*std_range)
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    result = np.clip(img.astype(np.float32) + noise, 0, 255)
    return result.astype(np.uint8)


def flip_and_rotate(img):
    """Kombinasi flip + rotasi."""
    return rotate(flip_horizontal(img))


def brightness_and_noise(img):
    """Kombinasi brightness + noise."""
    return add_gaussian_noise(adjust_brightness(img))


# Daftar semua teknik augmentasi yang tersedia
AUGMENT_TECHNIQUES = [
    ("flip",          flip_horizontal),
    ("rotate",        rotate),
    ("brightness",    adjust_brightness),
    ("noise",         add_gaussian_noise),
    ("flip_rotate",   flip_and_rotate),
    ("bright_noise",  brightness_and_noise),
]


# ─────────────────────────────────────────────
# PROSES AUGMENTASI PER KELAS
# ─────────────────────────────────────────────
def augment_class(class_name, input_dir, output_dir, target):
    """
    Augmentasi gambar satu kelas sampai mencapai jumlah target.
    Gambar asli disalin dulu, sisanya diaugmentasi.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Ambil semua file gambar
    img_files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ])
    n_original = len(img_files)
    n_needed   = target - n_original

    print(f"\n[{class_name}]")
    print(f"  Gambar asli  : {n_original}")
    print(f"  Target total : {target}")
    print(f"  Perlu tambah : {n_needed} gambar augmentasi")

    counter = 1

    # 1. Salin semua gambar asli ke output
    for fname in img_files:
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, f"orig_{counter:03d}.jpg")
        img = cv2.imread(src)
        cv2.imwrite(dst, img)
        counter += 1

    # 2. Buat gambar augmentasi sampai target tercapai
    aug_idx = 0
    while counter <= target:
        # Pilih gambar sumber secara acak
        src_file = random.choice(img_files)
        src_path = os.path.join(input_dir, src_file)
        img = cv2.imread(src_path)

        if img is None:
            continue

        # Pilih teknik augmentasi secara bergantian (tidak acak)
        # supaya distribusi teknik merata
        tech_name, tech_func = AUGMENT_TECHNIQUES[
            aug_idx % len(AUGMENT_TECHNIQUES)
        ]

        aug_img = tech_func(img)
        dst = os.path.join(output_dir,
                           f"aug_{tech_name}_{counter:03d}.jpg")
        cv2.imwrite(dst, aug_img)

        counter += 1
        aug_idx += 1

    print(f"  ✔ Total tersimpan: {counter - 1} gambar")
    print(f"    Teknik digunakan: "
          f"{', '.join([t[0] for t in AUGMENT_TECHNIQUES])}")

    return counter - 1


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  AUGMENTASI DATASET")
    print("=" * 55)
    print(f"  Input  : {INPUT_PATH}")
    print(f"  Output : {OUTPUT_PATH}")
    print(f"  Target : {TARGET_PER_CLASS} gambar per kelas")

    os.makedirs(OUTPUT_PATH, exist_ok=True)

    classes = sorted([
        d for d in os.listdir(INPUT_PATH)
        if os.path.isdir(os.path.join(INPUT_PATH, d))
    ])

    print(f"  Kelas  : {classes}")

    total_results = {}
    for class_name in classes:
        input_dir  = os.path.join(INPUT_PATH, class_name)
        output_dir = os.path.join(OUTPUT_PATH, class_name)

        total = augment_class(
            class_name, input_dir, output_dir,
            TARGET_PER_CLASS
        )
        total_results[class_name] = total

    # Ringkasan
    print("\n" + "=" * 55)
    print("  RINGKASAN HASIL AUGMENTASI")
    print("=" * 55)
    grand_total = 0
    for cls, count in total_results.items():
        print(f"  {cls:<15} : {count} gambar")
        grand_total += count
    print(f"  {'TOTAL':<15} : {grand_total} gambar")
    print("=" * 55)
    print(f"\nDataset augmentasi tersimpan di: '{OUTPUT_PATH}/'")
    print("   Selanjutnya ubah DATASET_PATH di mlp_klasifikasi.py")
    print(f"   menjadi: DATASET_PATH = \"{OUTPUT_PATH}\"")


if __name__ == "__main__":
    main()