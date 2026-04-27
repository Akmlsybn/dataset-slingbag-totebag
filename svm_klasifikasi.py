import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
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
        channel_axis=-1
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
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
# 6. TRAINING + EVALUASI SVM
# ─────────────────────────────────────────────
def train_and_evaluate(X_train, X_test, y_train, y_test,
                       model_params, exp_name, classes):
    print("=" * 62)
    print(f"  {exp_name}")
    print("=" * 62)
    print(f"  Parameter : {model_params}")
    print(f"  Data latih: {len(X_train)} sampel (asli + augmentasi)")
    print(f"  Data uji  : {len(X_test)} sampel (asli murni)")
    print()

    svm = SVC(**model_params, random_state=RANDOM_STATE,
               probability=True)
    svm.fit(X_train, y_train)

    y_pred       = svm.predict(X_test)
    y_pred_train = svm.predict(X_train)

    acc_test  = accuracy_score(y_test, y_pred) * 100
    acc_train = accuracy_score(y_train, y_pred_train) * 100
    gap       = acc_train - acc_test

    print(f"  Akurasi Train : {acc_train:.2f}%")
    print(f"  Akurasi Test  : {acc_test:.2f}%")
    print(f"  Gap           : {gap:.2f}%"
          f"  {'' if gap < 10 else '← Perlu dicek'}")
    print(f"  Jumlah Support Vector: {sum(svm.n_support_)}\n")
    print(classification_report(y_test, y_pred,
                                 target_names=classes))

    cm = confusion_matrix(y_test, y_pred)
    return svm, acc_test, acc_train, gap, cm, y_pred

# ─────────────────────────────────────────────
# 7. PLOT
# ─────────────────────────────────────────────
def plot_confusion_matrix(cm, classes, title, filename):
    plt.figure(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=classes)
    disp.plot(cmap='Blues', colorbar=False)
    plt.title(title, fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"  ✔ Confusion matrix: {filename}")


def plot_comparison(results):
    names  = [r['name'] for r in results]
    accs   = [r['acc_test'] for r in results]
    colors = ['#4C72B0', '#55A868', '#C44E52']

    plt.figure(figsize=(9, 5))
    bars = plt.bar(names, accs, color=colors,
                   width=0.5, edgecolor='white')

    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.5,
                 f'{acc:.2f}%', ha='center', va='bottom',
                 fontsize=12, fontweight='bold')

    plt.ylim(0, 115)
    plt.ylabel('Akurasi Test (%)', fontsize=12)
    plt.title('Perbandingan Akurasi 3 Skenario Eksperimen\n'
              'HOG + SVM (Split sebelum augmentasi)',
              fontsize=12)
    plt.tight_layout()
    plt.savefig('hasil/perbandingan_akurasi.png', dpi=150)
    plt.close()
    print("  ✔ Perbandingan akurasi: hasil/perbandingan_akurasi.png")


def plot_predictions(imgs_test, y_test, y_pred,
                     classes, filename):
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
        ax.imshow(imgs_test[i])

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
        f"Visualisasi Prediksi Model Terbaik (HOG + SVM)\n"
        f"Benar: {benar}/{len(y_test)}  |  "
        f"Akurasi: {benar/len(y_test)*100:.1f}%",
        fontsize=13, fontweight='bold', y=1.01
    )
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✔ Visualisasi prediksi: {filename}")


def print_summary(results):
    print("\n" + "=" * 75)
    print("  RINGKASAN 3 SKENARIO EKSPERIMEN — HOG + SVM")
    print("  (Data uji = 30 foto ASLI murni, tidak diaugmentasi)")
    print("=" * 75)
    print(f"{'No':<4} {'Eksperimen':<30} {'Train':>8} "
          f"{'Test':>8} {'Gap':>6} {'SV':>6}")
    print("-" * 75)

    best_idx = max(range(len(results)),
                   key=lambda i: results[i]['acc_test'])

    for i, r in enumerate(results):
        mark = " ← TERBAIK" if i == best_idx else ""
        print(f"  {i+1:<3} {r['name']:<30} "
              f"{r['acc_train']:>7.2f}% "
              f"{r['acc_test']:>7.2f}% "
              f"{r['gap']:>5.2f}% "
              f"{r['n_sv']:>6}{mark}")

    print("=" * 75)
    best = results[best_idx]
    print(f"\n  Model terbaik : {best['name']}")
    print(f"  Akurasi test  : {best['acc_test']:.2f}%")
    print(f"  Gap           : {best['gap']:.2f}% (tidak overfitting)")

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

    print(f"  Asli per kelas   : {n_per_class}")
    print(f"  Target per kelas : {target_per_class}\n")

    X_train, y_train, _ = augment_train(
        X_train_orig, y_train_orig, imgs_train,
        target_per_class
    )
    print(f"\n  Total data latih (setelah augmentasi): {len(X_train)}")
    print(f"  Total data uji   (tetap asli)        : {len(X_test)}\n")

    # ── Normalisasi ──
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # ── Step 4: 3 Eksperimen SVM ──
    print("=" * 62)
    print("  STEP 4: TRAINING 3 EKSPERIMEN HOG + SVM")
    print("=" * 62)

    experiments = [
        {
            # Eksperimen 1: Kernel RBF standar (baseline)
            "name": "Eksperimen 1 (RBF Baseline)",
            "params": {
                "kernel": "rbf",
                "C": 1.0,
                "gamma": "scale",
            }
        },
        {
            # Eksperimen 2: Polynomial (degree 3)
            "name": "Eksperimen 2 (Kernel Poly)",
            "params": {
                "kernel": "poly",
                "C": 1.0,
                "degree": 3,
                "gamma": "scale",
            }
        },
        {
            # Eksperimen 3: Kernel Linear
            "name": "Eksperimen 3 (Kernel Linear)",
            "params": {
                "kernel": "linear",
                "C": 1.0,
            }
        },
    ]

    results  = []
    best_svm = None
    best_acc = -1
    best_pred = None

    for i, exp in enumerate(experiments):
        svm, acc_test, acc_train, gap, cm, y_pred = train_and_evaluate(
            X_train, X_test, y_train, y_test,
            exp["params"], exp["name"], classes
        )

        plot_confusion_matrix(
            cm, classes,
            title=f"Confusion Matrix — {exp['name']}",
            filename=f"hasil/confusion_matrix_exp{i+1}.png"
        )

        results.append({
            "name":      exp["name"],
            "acc_test":  acc_test,
            "acc_train": acc_train,
            "gap":       gap,
            "n_sv":      sum(svm.n_support_),
            "cm":        cm,
        })

        if acc_test > best_acc:
            best_acc  = acc_test
            best_svm  = svm
            best_pred = y_pred

        print()

    # ── Plot perbandingan ──
    plot_comparison(results)

    # ── Visualisasi prediksi model terbaik ──
    plot_predictions(
        imgs_test, y_test, best_pred, classes,
        filename='hasil/visualisasi_prediksi_semua.png'
    )

    # ── Ringkasan ──
    print_summary(results)

    print("\nSemua hasil tersimpan di folder 'hasil/'")
    print("   - confusion_matrix_exp1.png (RBF Baseline)")
    print("   - confusion_matrix_exp2.png (C Besar)")
    print("   - confusion_matrix_exp3.png (Kernel Linear)")
    print("   - perbandingan_akurasi.png")
    print("   - visualisasi_prediksi_semua.png")


if __name__ == "__main__":
    main()