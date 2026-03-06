import os
import cv2

dataset_dir = 'Dataset Backpack & Totebag' 

kategori = ['Ransel', 'Totebag'] 

ukuran_gambar = (256, 256) 

folder_simpan = 'Dataset_Preprocessed'

if not os.path.exists(folder_simpan):
    os.makedirs(folder_simpan)

total_diproses = 0

for label in kategori:
    path_baca = os.path.join(dataset_dir, label)
    path_simpan_sub = os.path.join(folder_simpan, label)
    
    if not os.path.exists(path_simpan_sub):
        os.makedirs(path_simpan_sub)
        
    for img_name in os.listdir(path_baca):
        img_path = os.path.join(path_baca, img_name)
        
        img_array = cv2.imread(img_path)
        
        if img_array is not None:
            img_resized = cv2.resize(img_array, ukuran_gambar)
            
            nama_file_baru = os.path.join(path_simpan_sub, img_name)

            cv2.imwrite(nama_file_baru, img_resized)
            total_diproses += 1
        else:
            print(f"error image {img_name}")