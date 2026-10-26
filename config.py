"""Konfigurasi bersama preprocessing, augmentasi, dan GLCM.

Letakkan semua file .py ini di folder yang sama dengan `dataset_gabung_fix/`
dan `eda_output/` (mis. D:\\Kelompok Machine Learning\\).
"""
from pathlib import Path

BASE_DIR    = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset_gabung_fix"
EDA_DIR     = BASE_DIR / "eda_output"            # split_train/val/test.csv dari EDA.py
OUT_DIR     = BASE_DIR / "preprocessing_output"

SEED     = 42
IMG_SIZE = 224                                   # input ResNet50 & MobileNetV2 (Bab 4)

# --- Preprocessing citra ---------------------------------------------------
CROP_MARGIN    = 0.15    # margin di sekitar kotak karakter (proporsi sisi terpanjang);
                         # memberi ruang agar augmentasi tidak memotong rarangkén
MIN_BLOB_FRAC  = 0.0005  # komponen terhubung lebih kecil dari ini (x luas citra) dianggap noise
DUP_THRESHOLD  = 0.95    # korelasi piksel (48x48) >= ini dianggap citra hampir-duplikat

# --- GLCM (konfigurasi awal, belum dioptimasi) -----------------------------
GLCM_LEVELS     = 32                 # kuantisasi tingkat keabuan 256 -> 32
GLCM_DISTANCES  = [1, 2]
GLCM_ANGLES_DEG = [0, 45, 90, 135]
GLCM_PROPS      = ["contrast", "homogeneity", "energy", "correlation"]
# 4 properti x 2 jarak x 4 sudut = 32 fitur

# --- Augmentasi (hanya data latih) -----------------------------------------
AUG_ROTATION_DEG = 10            # rotasi acak +-10 derajat
AUG_ZOOM         = (0.90, 1.05)  # skala acak (zoom-in dibatasi agar goresan tidak terpotong)
AUG_SHIFT        = 0.05          # geser acak +-5% sisi citra
AUG_NOISE_STD    = 0.0           # noise Gaussian (0 = mati); GLCM peka terhadap noise
# Tanpa flip: pencerminan mengubah bentuk aksara.

BATCH_SIZE = 32
