import os
import cv2

# 1. Sesuaikan dengan nama folder mentah kamu yang baru
dataset_dir = 'Dataset Slingbag & Totebag' 

# 2. Sesuaikan kategori dengan nama folder di dalam (slingbag huruf kecil, Totebag huruf depan besar)
kategori = ['slingbag', 'Totebag'] 

ukuran_gambar = (256, 256) 

# 3. Folder tujuan tetap sama
folder_simpan = 'Dataset_Preprocessed'

if not os.path.exists(folder_simpan):
    os.makedirs(folder_simpan)

total_diproses = 0

for label in kategori:
    path_baca = os.path.join(dataset_dir, label)
    path_simpan_sub = os.path.join(folder_simpan, label)
    
    if not os.path.exists(path_simpan_sub):
        os.makedirs(path_simpan_sub)
        
    print(f"Sedang memproses kategori: {label}")
        
    for img_name in os.listdir(path_baca):
        img_path = os.path.join(path_baca, img_name)
        
        # Membaca gambar
        img_array = cv2.imread(img_path)
        
        if img_array is not None:
            # Resize gambar
            img_resized = cv2.resize(img_array, ukuran_gambar)
            
            # Simpan dengan nama file yang sama
            nama_file_baru = os.path.join(path_simpan_sub, img_name)

            cv2.imwrite(nama_file_baru, img_resized)
            total_diproses += 1
        else:
            print(f"Gagal membaca gambar: {img_name}")

print(f"\nSelesai! Total {total_diproses} gambar telah di-resize ke 256x256.")