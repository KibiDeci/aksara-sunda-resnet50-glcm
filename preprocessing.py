"""Preprocessing citra karakter Aksara Sunda (Bab 4 laporan).

Urutan: baca -> grayscale -> samakan polaritas (goresan gelap, latar terang)
-> potong area karakter -> padding persegi (proporsi terjaga) -> resize 224x224.

Hasilnya satu citra grayscale uint8 yang menjadi sumber KEDUA cabang:
  * cabang GLCM  : memakai citra grayscale ini langsung (sebelum normalisasi ImageNet)
  * cabang CNN   : citra disalin ke 3 kanal lalu dinormalisasi sesuai arsitektur
"""
from pathlib import Path, PureWindowsPath

import cv2
import numpy as np
from PIL import Image, ImageOps

import config as C


# ---------------------------------------------------------------------------
# Langkah-langkah dasar
# ---------------------------------------------------------------------------
def load_gray(path) -> np.ndarray:
    """Baca citra apa pun (RGB / L) menjadi grayscale uint8, orientasi EXIF dibetulkan."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        return np.asarray(im.convert("L"), dtype=np.uint8)


def border_value(gray: np.ndarray) -> int:
    """Perkiraan warna latar: median piksel tepi citra."""
    edge = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    return int(np.median(edge))


def ensure_dark_on_light(gray: np.ndarray) -> np.ndarray:
    """Jika latar gelap (goresan terang), balik intensitas agar seragam."""
    return 255 - gray if border_value(gray) < 127 else gray


def character_bbox(gray: np.ndarray):
    """Kotak pembatas goresan karakter (x0, y0, x1, y1) via threshold Otsu.

    Mengembalikan None bila tidak ada goresan yang terdeteksi.
    """
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    min_area = C.MIN_BLOB_FRAC * gray.size
    keep = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not keep:
        return None
    ys, xs = np.nonzero(np.isin(lab, keep))
    return xs.min(), ys.min(), xs.max() + 1, ys.max() + 1


def crop_character(gray: np.ndarray, margin: float = C.CROP_MARGIN):
    """Potong area karakter + margin. Mengembalikan (citra, berhasil_crop)."""
    box = character_bbox(gray)
    if box is None:
        return gray, False
    x0, y0, x1, y1 = box
    m = int(round(margin * max(x1 - x0, y1 - y0)))
    h, w = gray.shape
    return gray[max(0, y0 - m):min(h, y1 + m), max(0, x0 - m):min(w, x1 + m)], True


def pad_to_square(gray: np.ndarray, value: int) -> np.ndarray:
    """Tambahkan padding (warna latar) agar persegi tanpa meregangkan karakter."""
    h, w = gray.shape
    s = max(h, w)
    top, left = (s - h) // 2, (s - w) // 2
    return cv2.copyMakeBorder(gray, top, s - h - top, left, s - w - left,
                              cv2.BORDER_CONSTANT, value=int(value))


def resize(gray: np.ndarray, size: int = C.IMG_SIZE) -> np.ndarray:
    interp = cv2.INTER_AREA if gray.shape[0] > size else cv2.INTER_CUBIC
    return cv2.resize(gray, (size, size), interpolation=interp)


# ---------------------------------------------------------------------------
# Pipeline lengkap
# ---------------------------------------------------------------------------
def preprocess_gray(path, return_info: bool = False):
    """Citra mentah -> grayscale 224x224 uint8 yang siap untuk CNN & GLCM."""
    gray = ensure_dark_on_light(load_gray(path))
    bg = border_value(gray)
    crop, ok = crop_character(gray)
    out = resize(pad_to_square(crop, bg))
    if return_info:
        return out, {"crop_ok": ok, "crop_h": crop.shape[0], "crop_w": crop.shape[1],
                     "orig_h": gray.shape[0], "orig_w": gray.shape[1], "bg": bg}
    return out


def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
    """Salin grayscale ke 3 kanal (float32, rentang 0-255) untuk bobot ImageNet."""
    return np.repeat(gray[..., None], 3, axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# Normalisasi khusus arsitektur (identik dengan keras.applications.*.preprocess_input)
# ---------------------------------------------------------------------------
_IMAGENET_BGR_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def normalize_for(arch: str, rgb: np.ndarray) -> np.ndarray:
    """rgb float32 0-255 (..., H, W, 3) -> input yang diharapkan model pralatih.

    resnet50    : mode 'caffe'  -> RGB ke BGR lalu dikurangi rata-rata ImageNet
    mobilenetv2 : mode 'tf'     -> diskalakan ke rentang [-1, 1]
    """
    arch = arch.lower()
    if arch == "resnet50":
        return rgb[..., ::-1] - _IMAGENET_BGR_MEAN
    if arch == "mobilenetv2":
        return rgb / 127.5 - 1.0
    raise ValueError(f"Arsitektur tidak dikenal: {arch}")


# ---------------------------------------------------------------------------
# Utilitas path (CSV split dari EDA berisi path absolut Windows D:\...)
# ---------------------------------------------------------------------------
def to_local_path(p: str) -> Path:
    """'D:\\...\\dataset_gabung_fix\\ka\\x.jpg' -> DATASET_DIR/ka/x.jpg (lintas OS)."""
    parts = PureWindowsPath(p).parts if "\\" in p else Path(p).parts
    return C.DATASET_DIR / parts[-2] / parts[-1]
