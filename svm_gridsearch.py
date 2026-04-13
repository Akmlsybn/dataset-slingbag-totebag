"""
Klasifikasi Fine-Grained Slingbag vs Totebag
Menggunakan HOG + SVM dengan GRID SEARCH (K-FOLD CV)
ALUR BENAR: Split dulu → Augmentasi hanya data latih (anti data leakage)
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV  # <-- LIBRARY GRID SEARCH
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay)
import random
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────────
DATASET_PATH = "Dataset_Preprocessed"
IMG_SIZE     = 256
RANDOM_STATE = 42
TEST_SIZE    = 0.4

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────
# 1. FUNGSI AUGMENTASI (OpenCV — ML Tradisional)
# ─────────────────────────────────────────────
def flip_horizontal(img):
    return cv2.flip(img, 1)

def rotate(img, angle_range=(-15, 15)):
    angle  = random.uniform(*angle_range)
    h, w   = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT)

def adjust_brightness(img, factor_range=(0.6, 1.4)):
    factor = random.uniform(*factor_range)
    return np.clip(img.astype(np.float32) * factor,
                   0, 255).astype(np.uint8)

def add_gaussian_noise(img, std_range=(5, 20)):
    std   = random.uniform(*std_range)
    noise = np.random.normal(0, std, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise,
                   0, 255).astype(np.uint8)

def flip_and_rotate(img):
    return rotate(flip_horizontal(img))

def brightness_and_noise(img):
    return add_gaussian_noise(adjust_brightness(img))

AUGMENT_FUNCS = [
    flip_horizontal,
    rotate,
    adjust_brightness,
    add_gaussian_noise,
    flip_and_rotate,
    brightness_and_noise,
]

# ─────────────────────────────────────────────
# 2. EKSTRAKSI FITUR HOG
# ─────────────────────────────────────────────
def extract_hog(img):
    """Ekstraksi fitur HOG langsung dari gambar BGR/RGB."""
    return hog(
        img,
        orientations=9,
        pixels_per_cell=(64, 64),
        cells_per_block=(2, 2),
        block_norm='L2-Hys',
        channel_axis=-1  # <- Ini yang bikin aman 324 dimensi dari RGB
    )

# ─────────────────────────────────────────────
# 3. LOAD DATASET ASLI
# ─────────────────────────────────────────────
def load_original_dataset(dataset_path, img_size=256):
    X, y, imgs_raw = [], [], []
    classes = sorted(os.listdir(dataset_path))

    print(f"Kelas ditemukan: {classes}\n")

    for label, class_name in enumerate(classes):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        img_files = sorted([
            f for f in os.listdir(class_path)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        print(f"[{class_name}] — {len(img_files)} gambar asli")

        for fname in img_files:
            img = cv2.imread(os.path.join(class_path, fname))
            if img is None:
                continue
            img = cv2.resize(img, (img_size, img_size))
            X.append(extract_hog(img))
            y.append(label)
            imgs_raw.append(img)

    return np.array(X), np.array(y), imgs_raw, classes

# ─────────────────────────────────────────────
# 4. SPLIT STRATIFIED (SEBELUM AUGMENTASI)
# ─────────────────────────────────────────────
def stratified_split(X, y, imgs, test_size=0.4, random_state=42):
    rng     = np.random.RandomState(random_state)
    classes = np.unique(y)

    train_idx, test_idx = [], []

    for cls in classes:
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        n_test = int(len(idx) * test_size)
        test_idx.extend(idx[:n_test])
        train_idx.extend(idx[n_test:])

    train_idx = np.array(train_idx)
    test_idx  = np.array(test_idx)

    return (X[train_idx], X[test_idx],
            y[train_idx], y[test_idx],
            [imgs[i] for i in train_idx],
            [imgs[i] for i in test_idx])

# ─────────────────────────────────────────────
# 5. AUGMENTASI HANYA DATA LATIH
# ─────────────────────────────────────────────
def augment_train(X_train, y_train, imgs_train, target_per_class):
    X_aug   = list(X_train)
    y_aug   = list(y_train)
    imgs_aug = list(imgs_train)

    for cls in np.unique(y_train):
        cls_idx = [i for i, lbl in enumerate(y_train) if lbl == cls]
        needed  = target_per_class - len(cls_idx)

        aug_counter = 0
        while aug_counter < needed:
            src_img = imgs_train[random.choice(cls_idx)]
            func    = AUGMENT_FUNCS[aug_counter % len(AUGMENT_FUNCS)]
            aug_img = func(src_img)

            X_aug.append(extract_hog(aug_img))
            y_aug.append(cls)
            imgs_aug.append(aug_img)
            aug_counter += 1

    return np.array(X_aug), np.array(y_aug), imgs_aug

# ─────────────────────────────────────────────
# 6. VISUALISASI
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=classes)
    disp.plot(cmap='Blues', colorbar=False)
    plt.title(title, fontsize=11, pad=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  ✔ Confusion matrix: {filename}")


def plot_predictions(imgs_test, y_test, y_pred, classes, filename):
    n    = min(len(imgs_test), 30)
    cols = 5
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 3.2, rows * 3.8))
    axes = axes.flatten()

    for i in range(n):
        ax         = axes[i]
        true_label = classes[y_test[i]]
        pred_label = classes[y_pred[i]]
        correct    = y_test[i] == y_pred[i]
        color      = '#2ecc71' if correct else '#e74c3c'
        mark       = '✓' if correct else '✗'

        img_rgb = cv2.cvtColor(imgs_test[i], cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)

        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(4)

        ax.set_title(
            f"Asli: {true_label}\nPred: {pred_label} {mark}",
            fontsize=9, color=color,
            fontweight='bold', pad=4
        )
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    patch_benar = mpatches.Patch(color='#2ecc71', label='Benar ✓')
    patch_salah = mpatches.Patch(color='#e74c3c', label='Salah ✗')
    fig.legend(handles=[patch_benar, patch_salah],
               loc='lower center', ncol=2, fontsize=11,
               bbox_to_anchor=(0.5, 0.01))

    benar = sum(1 for a, b in zip(y_test, y_pred) if a == b)
    fig.suptitle(
        f"Visualisasi Prediksi Model Terbaik (Grid Search)\n"
        f"Benar: {benar}/{len(y_test)}  |  "
        f"Akurasi: {benar/len(y_test)*100:.1f}%",
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✔ Visualisasi prediksi: {filename}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    os.makedirs('hasil', exist_ok=True)

    # ── Step 1: Load data asli ──
    print("=" * 62)
    print("  STEP 1: LOAD DATASET ASLI (100 gambar)")
    print("=" * 62)
    X, y, imgs, classes = load_original_dataset(DATASET_PATH)
    print(f"\nTotal      : {len(X)} gambar asli")
    print(f"Fitur HOG  : {X.shape[1]} dimensi\n")

    # ── Step 2: Split DULU sebelum augmentasi ──
    print("=" * 62)
    print("  STEP 2: SPLIT 70:30 (SEBELUM AUGMENTASI)")
    print("=" * 62)
    (X_train_orig, X_test,
     y_train_orig, y_test,
     imgs_train, imgs_test) = stratified_split(
        X, y, imgs,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    print(f"\n  Data latih (asli) : {len(X_train_orig)} gambar")
    print(f"  Data uji (murni)  : {len(X_test)} gambar")
    print(f"  → Data uji TIDAK akan diaugmentasi\n")

    # ── Step 3: Augmentasi HANYA data latih ──
    print("=" * 62)
    print("  STEP 3: AUGMENTASI DATA LATIH SAJA")
    print("=" * 62)
    n_per_class   = len(X_train_orig) // len(np.unique(y_train_orig))
    target_per_class = n_per_class * 2

    X_train, y_train, _ = augment_train(
        X_train_orig, y_train_orig, imgs_train,
        target_per_class
    )
    print(f"  Total data latih (setelah augmentasi): {len(X_train)}")
    print(f"  Total data uji   (tetap asli)        : {len(X_test)}\n")

    # ── Normalisasi ──
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Step 4: IMPLEMENTASI GRID SEARCH (K-FOLD CV) ──
    print("=" * 62)
    print("  STEP 4: TRAINING DENGAN GRID SEARCH (5-FOLD CV)")
    print("=" * 62)

    # Menyiapkan 'keranjang' parameter untuk diuji otomatis
    param_grid = [
        {'kernel': ['linear'], 'C': [0.1, 1.0, 10.0]},
        {'kernel': ['rbf'], 'C': [0.1, 1.0, 10.0], 'gamma': ['scale', 'auto']},
        {'kernel': ['poly'], 'C': [0.1, 1.0, 10.0], 'degree': [2, 3], 'gamma': ['scale']}
    ]

    print("Sedang mencari kombinasi parameter terbaik...")
    print("Mengevaluasi berbagai kombinasi menggunakan 5-Fold Cross Validation...\n")

    grid = GridSearchCV(
        SVC(probability=True, random_state=RANDOM_STATE),
        param_grid,
        cv=5,               # <--- K-Fold Cross Validation (5 lipatan)
        scoring='accuracy',
        verbose=1,          # Menampilkan log progress
        n_jobs=-1           # Menggunakan seluruh core CPU agar lebih cepat
    )

    grid.fit(X_train, y_train)
    best_svm = grid.best_estimator_

    print("\n" + "=" * 62)
    print("  ✅ PENCARIAN GRID SEARCH SELESAI!")
    print("=" * 62)
    print(f"  Parameter Terbaik  : {grid.best_params_}")
    print(f"  Akurasi K-Fold CV  : {grid.best_score_ * 100:.2f}% (Rata-rata saat validasi silang)\n")

    # ── Step 5: EVALUASI MODEL TERBAIK PADA DATA UJI MURNI ──
    print("=" * 62)
    print("  STEP 5: EVALUASI PADA 30 DATA UJI MURNI")
    print("=" * 62)

    y_pred_train = best_svm.predict(X_train)
    y_pred_test  = best_svm.predict(X_test)

    acc_train = accuracy_score(y_train, y_pred_train) * 100
    acc_test  = accuracy_score(y_test, y_pred_test) * 100
    gap       = acc_train - acc_test

    print(f"  Akurasi Train (Full) : {acc_train:.2f}%")
    print(f"  Akurasi Test (Murni) : {acc_test:.2f}%")
    print(f"  Gap                  : {gap:.2f}% "
          f"{'← OK (tidak overfitting)' if gap < 10 else '← Perlu dicek'}")
    print(f"  Jumlah Support Vector: {sum(best_svm.n_support_)}\n")

    print("  Classification Report (Data Uji):")
    print(classification_report(y_test, y_pred_test, target_names=classes))

    # ── Visualisasi Akhir ──
    cm = confusion_matrix(y_test, y_pred_test)
    plot_confusion_matrix(
        cm, classes,
        title=f"Confusion Matrix (Grid Search Best Model)\n{grid.best_params_}",
        filename="hasil/confusion_matrix_gridsearch.png"
    )

    plot_predictions(
        imgs_test, y_test, y_pred_test, classes,
        filename='hasil/visualisasi_prediksi_gridsearch.png'
    )

    print("\n✅ Proses selesai! Gambar tersimpan di folder 'hasil/'")

if __name__ == "__main__":
    main()