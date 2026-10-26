"""Penyedia batch untuk pelatihan M1-M4 (Keras).

  M1 ResNet50          : AksaraDataset(..., arch="resnet50",    with_glcm=False)
  M2 ResNet50-GLCM     : AksaraDataset(..., arch="resnet50",    with_glcm=True)
  M3 MobileNetV2       : AksaraDataset(..., arch="mobilenetv2", with_glcm=False)
  M4 MobileNetV2-GLCM  : AksaraDataset(..., arch="mobilenetv2", with_glcm=True)

Keluaran per batch:
  tanpa GLCM : (citra, label)
  dengan GLCM: ((citra, fitur_glcm), label)   -> model 2 input
Label berupa bilangan bulat (sparse categorical cross-entropy).

Contoh:
    import pandas as pd, joblib, json
    from data_loader import AksaraDataset, load_split, load_cache
    cache  = load_cache()                                   # citra 224x224 hasil run_preprocessing.py
    scaler = joblib.load("preprocessing_output/scaler_glcm.joblib")
    classes = json.load(open("preprocessing_output/class_names.json"))
    train = AksaraDataset(load_split("train"), classes, cache, arch="resnet50",
                          with_glcm=True, scaler=scaler, augment=True, shuffle=True)
    val   = AksaraDataset(load_split("val"),   classes, cache, arch="resnet50",
                          with_glcm=True, scaler=scaler)
    model.fit(train, validation_data=val, epochs=...)
"""
import math

import numpy as np
import pandas as pd

import config as C
from augmentation import augment_gray
from glcm_features import glcm_features
from preprocessing import gray_to_rgb, normalize_for, preprocess_gray

try:                                    # Keras 3 / TF >= 2.16
    from keras.utils import PyDataset as _Base
except ImportError:                     # TF lama
    from tensorflow.keras.utils import Sequence as _Base


def load_split(name: str) -> pd.DataFrame:
    """Baca split hasil run_preprocessing.py (kolom: relpath, label, source)."""
    return pd.read_csv(C.OUT_DIR / f"split_{name}.csv")


def load_cache() -> dict:
    """{relpath: citra grayscale 224x224} dari preprocessed_224.npz."""
    z = np.load(C.OUT_DIR / "preprocessed_224.npz")
    return dict(zip(z["relpath"].tolist(), z["images"]))


class AksaraDataset(_Base):
    def __init__(self, df, class_names, cache=None, arch="resnet50", with_glcm=False,
                 scaler=None, augment=False, shuffle=False, batch_size=C.BATCH_SIZE,
                 seed=C.SEED, **kwargs):
        super().__init__(**kwargs)
        if with_glcm and scaler is None:
            raise ValueError("with_glcm=True butuh scaler yang di-fit pada data latih.")
        self.df = df.reset_index(drop=True)
        self.cls_idx = {c: i for i, c in enumerate(class_names)}
        self.y = self.df.label.map(self.cls_idx).to_numpy(np.int32)
        self.cache, self.arch, self.with_glcm = cache or {}, arch, with_glcm
        self.scaler, self.augment, self.shuffle = scaler, augment, shuffle
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(self.df))
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return math.ceil(len(self.df) / self.batch_size)

    def _gray(self, i):
        rel = self.df.relpath[i]
        g = self.cache.get(rel)
        return g if g is not None else preprocess_gray(C.DATASET_DIR / rel)

    def __getitem__(self, b):
        idx = self.order[b * self.batch_size:(b + 1) * self.batch_size]
        grays = [self._gray(i) for i in idx]
        if self.augment:
            grays = [augment_gray(g, self.rng) for g in grays]

        x_img = normalize_for(self.arch, np.stack([gray_to_rgb(g) for g in grays]))
        y = self.y[idx]
        if not self.with_glcm:
            return x_img, y
        x_glcm = self.scaler.transform(np.stack([glcm_features(g) for g in grays]))
        return (x_img, x_glcm.astype(np.float32)), y

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)
