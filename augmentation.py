"""Augmentasi data latih.

Augmentasi dikenakan pada citra grayscale 224x224 hasil preprocessing, SEBELUM
dipecah ke cabang CNN dan GLCM, sehingga fitur GLCM dihitung dari citra yang
mengalami transformasi yang sama dengan input CNN (Bab 4 laporan).
Data validasi dan uji tidak pernah diaugmentasi.
"""
import cv2
import numpy as np

import config as C
from preprocessing import border_value


def augment_gray(gray: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    h, w = gray.shape
    bg = border_value(gray)

    angle = rng.uniform(-C.AUG_ROTATION_DEG, C.AUG_ROTATION_DEG)
    scale = rng.uniform(*C.AUG_ZOOM)
    tx, ty = rng.uniform(-C.AUG_SHIFT, C.AUG_SHIFT, size=2) * (w, h)

    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    M[:, 2] += (tx, ty)
    out = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=int(bg))

    if C.AUG_NOISE_STD > 0:
        noise = rng.normal(0, C.AUG_NOISE_STD, out.shape)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out
