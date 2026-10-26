"""Ekstraksi fitur tekstur GLCM: 4 properti x 2 jarak x 4 sudut = 32 fitur.

Dihitung dari citra grayscale 224x224 (sebelum normalisasi ImageNet),
setelah dikuantisasi ke GLCM_LEVELS tingkat keabuan.
"""
import numpy as np
from skimage.feature import graycomatrix, graycoprops

import config as C

_ANGLES = np.deg2rad(C.GLCM_ANGLES_DEG)


def quantize(gray: np.ndarray, levels: int = C.GLCM_LEVELS) -> np.ndarray:
    return (gray.astype(np.uint16) * levels // 256).astype(np.uint8)


def glcm_features(gray: np.ndarray) -> np.ndarray:
    P = graycomatrix(quantize(gray), distances=C.GLCM_DISTANCES, angles=_ANGLES,
                     levels=C.GLCM_LEVELS, symmetric=True, normed=True)
    feats = np.concatenate([graycoprops(P, prop).ravel() for prop in C.GLCM_PROPS])
    # correlation tak terdefinisi (NaN) bila citra seragam -> 0
    return np.nan_to_num(feats, nan=0.0).astype(np.float32)


def feature_names():
    return [f"{p}_d{d}_a{a}" for p in C.GLCM_PROPS
            for d in C.GLCM_DISTANCES for a in C.GLCM_ANGLES_DEG]


N_FEATURES = len(C.GLCM_PROPS) * len(C.GLCM_DISTANCES) * len(C.GLCM_ANGLES_DEG)
