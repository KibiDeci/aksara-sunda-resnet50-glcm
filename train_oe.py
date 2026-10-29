"""M2-OE: M2 (ResNet50-GLCM) yang dilatih lanjut dengan Outlier Exposure.

    python buat_outlier.py     # sekali saja: membuat contoh coretan non-aksara di outlier/
    python train_oe.py         # butuh hasil_M2/best.keras (latih M2 dulu)

Masalah yang diatasi
    Model M1-M4 hanya mengenal 30 kelas (closed-set). Coretan yang bukan aksara
    tetap dipaksa masuk ke salah satu kelas, sering dengan persentase tinggi.

Cara kerja (Hendrycks dkk., 2019)
    Model M2 dilatih lanjut dengan dua jenis data dalam setiap batch:
      - citra aksara  -> target one-hot kelas sebenarnya (seperti biasa)
      - citra coretan -> target seragam 1/30 untuk semua kelas
    Loss = CE(aksara) + LAMBDA * CE(coretan, seragam)
    Akibatnya, untuk coretan model belajar memberi persentase kecil yang merata
    (sekitar 3% per kelas), sementara aksara tetap diberi persentase tinggi.
    Softmax dan 30 kelas tetap sama; tidak ada kelas "bukan aksara".

Ambang "tidak dikenali" ditentukan dari data validasi aksara: persentil ke-5
probabilitas tertinggi, sehingga 95% aksara asli tetap diterima (standar FPR95).

Hasil di hasil_M2OE/:
    best.keras, history.csv, kurva_pelatihan.png, ambang_ood.json,
    ringkasan_hasil.json, tabel_oe.csv, grafik_oe.png, contoh_coretan_oe.png
M1-M4 dan hasil_M2/ tidak diubah.
"""
import argparse
import json
import math
import time

import cv2
import joblib
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import ops
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

import config as C
from augmentation import augment_gray
from data_loader import load_cache, load_split
from glcm_features import glcm_features
from preprocessing import gray_to_rgb, normalize_for, preprocess_gray

try:
    from keras.utils import PyDataset as _Base
except ImportError:
    from tensorflow.keras.utils import Sequence as _Base

ASAL, ARCH, GLCM = "M2", "resnet50", True
OUT = C.BASE_DIR / "hasil_M2OE"
OUTLIER_DIR = C.BASE_DIR / "outlier"
P = dict(lambda_oe=0.5, batch_aksara=32, batch_coretan=32,
         lr_tahap1=1e-4, lr_tahap2=1e-5, epochs_tahap1=10, epochs_tahap2=10,
         unfreeze_last=30, patience=3, persentil_ambang=5)


# ---------------------------------------------------------------- data
def muat_coretan(split):
    """Citra coretan setelah preprocessing yang sama dengan aksara (disimpan sebagai cache)."""
    cache = OUTLIER_DIR / f"outlier_{split}_224.npz"
    files = sorted((OUTLIER_DIR / split).glob("*.png"))
    if not files:
        raise SystemExit(f"{OUTLIER_DIR / split} kosong. Jalankan dulu: python buat_outlier.py")
    if cache.exists():
        z = np.load(cache)
        if len(z["images"]) == len(files):
            return list(z["images"])
    print(f"Preprocessing {len(files)} citra coretan ({split}) ...")
    g = np.stack([preprocess_gray(f) for f in files])
    np.savez_compressed(cache, images=g)
    return list(g)


def muat_aksara(split, cache, idx):
    df = load_split(split)
    grays = [cache[r] if r in cache else preprocess_gray(C.DATASET_DIR / r) for r in df.relpath]
    return grays, np.array([idx[l] for l in df.label], np.int32)


def ke_input(grays, scaler):
    x = normalize_for(ARCH, np.stack([gray_to_rgb(g) for g in grays]))
    if not GLCM:
        return x
    f = scaler.transform(np.stack([glcm_features(g) for g in grays])).astype(np.float32)
    return (x, f)


class OEDataset(_Base):
    """Setiap batch: aksara (target one-hot, bobot 1) + coretan (target 1/30, bobot LAMBDA)."""

    def __init__(self, ak_grays, ak_y, co_grays, n_kelas, scaler, augment=False, shuffle=False,
                 seed=C.SEED, **kw):
        super().__init__(**kw)
        self.ak, self.y, self.co = ak_grays, ak_y, co_grays
        self.n, self.scaler, self.augment, self.shuffle = n_kelas, scaler, augment, shuffle
        self.rng = np.random.default_rng(seed)
        self.o_ak, self.o_co = np.arange(len(ak_grays)), np.arange(len(co_grays))
        self.k_co = P["batch_coretan"] if shuffle else math.ceil(len(co_grays) / len(self))
        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.ak) / P["batch_aksara"])

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.o_ak); self.rng.shuffle(self.o_co)

    def __getitem__(self, b):
        ia = self.o_ak[b * P["batch_aksara"]:(b + 1) * P["batch_aksara"]]
        ic = np.take(self.o_co, range(b * self.k_co, (b + 1) * self.k_co), mode="wrap")
        if not self.shuffle:                                  # validasi: tiap coretan dipakai sekali
            ic = self.o_co[b * self.k_co:(b + 1) * self.k_co]
        grays = [self.ak[i] for i in ia] + [self.co[i] for i in ic]
        if self.augment:
            grays = [augment_gray(g, self.rng) for g in grays]
        y = np.concatenate([np.eye(self.n, dtype=np.float32)[self.y[ia]],
                            np.full((len(ic), self.n), 1 / self.n, np.float32)])
        w = np.concatenate([np.ones(len(ia), np.float32),
                            np.full(len(ic), P["lambda_oe"], np.float32)])
        return ke_input(grays, self.scaler), y, w


class AkurasiAksara(keras.metrics.Metric):
    """Akurasi yang hanya dihitung pada citra aksara (coretan tidak punya kelas benar)."""

    def __init__(self, name="akurasi_aksara", **kw):
        super().__init__(name=name, **kw)
        self.benar = self.add_variable(shape=(), initializer="zeros", name="benar")
        self.total = self.add_variable(shape=(), initializer="zeros", name="total")

    def update_state(self, y_true, y_pred, sample_weight=None):
        m = ops.cast(ops.max(y_true, axis=-1) > 0.5, "float32")
        c = ops.cast(ops.argmax(y_true, axis=-1) == ops.argmax(y_pred, axis=-1), "float32")
        self.benar.assign(self.benar + ops.sum(c * m)); self.total.assign(self.total + ops.sum(m))

    def result(self):
        return self.benar / ops.maximum(self.total, 1.0)

    def reset_state(self):
        self.benar.assign(0.0); self.total.assign(0.0)


# ---------------------------------------------------------------- evaluasi
def prob(model, grays, scaler, bs=32):
    return np.concatenate([model.predict(ke_input(grays[i:i + bs], scaler), verbose=0)
                           for i in range(0, len(grays), bs)])


def nilai(nama, model, data, scaler):
    pv, pt, pc = (prob(model, data[k], scaler) for k in ("val_ak", "test_ak", "test_co"))
    yt = data["test_y"]
    ambang = float(np.percentile(pv.max(1), P["persentil_ambang"]))
    pred = pt.argmax(1)
    return dict(model=nama,
                akurasi_uji_aksara=round(accuracy_score(yt, pred), 4),
                f1_macro_uji_aksara=round(f1_score(yt, pred, average="macro"), 4),
                ambang_probabilitas=round(ambang, 4),
                aksara_uji_diterima=round(float((pt.max(1) >= ambang).mean()), 4),
                coretan_uji_ditolak=round(float((pc.max(1) < ambang).mean()), 4),
                auroc=round(roc_auc_score(np.r_[np.ones(len(pt)), np.zeros(len(pc))],
                                          np.r_[pt.max(1), pc.max(1)]), 4),
                rerata_prob_tertinggi_aksara=round(float(pt.max(1).mean()), 4),
                rerata_prob_tertinggi_coretan=round(float(pc.max(1).mean()), 4)), pt, pc


def grafik(hasil, path):
    fig, axs = plt.subplots(1, len(hasil), figsize=(5.5 * len(hasil), 3.8), squeeze=False)
    bins = np.linspace(0, 1, 41)
    for ax, (r, pt, pc) in zip(axs[0], hasil):
        ax.hist(pt.max(1), bins, alpha=.7, label="aksara (uji)", color="#2b6cb0")
        ax.hist(pc.max(1), bins, alpha=.7, label="coretan (uji)", color="#dd6b20")
        ax.axvline(r["ambang_probabilitas"], color="k", ls="--", lw=1)
        ax.set(title=f"{r['model']}  (AUROC {r['auroc']:.3f})", yscale="log",
               xlabel="Probabilitas tertinggi (softmax)", ylabel="Jumlah citra (log)")
        ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def contoh_coretan(model, grays, scaler, classes, path, n=16):
    p = prob(model, grays[:n], scaler)
    fig, axs = plt.subplots(2, n // 2, figsize=(n * .95, 4.4))
    for ax, g, q in zip(axs.ravel(), grays, p):
        ax.imshow(g, cmap="gray", vmin=0, vmax=255); ax.axis("off")
        ax.set_title(f"{classes[q.argmax()]} {q.max() * 100:.0f}%", fontsize=9)
    fig.suptitle("M2-OE pada coretan uji (kelas teratas dan probabilitasnya)")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


# ---------------------------------------------------------------- utama
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="uji cepat dengan data sedikit")
    a = ap.parse_args()
    keras.utils.set_random_seed(C.SEED)
    OUT.mkdir(exist_ok=True)
    src = C.BASE_DIR / f"hasil_{ASAL}" / "best.keras"
    if not src.exists():
        raise SystemExit(f"{src} tidak ditemukan. Latih dulu: python train_model.py --model {ASAL}")

    classes = json.load(open(C.OUT_DIR / "class_names.json"))
    idx = {c: i for i, c in enumerate(classes)}
    scaler = joblib.load(C.OUT_DIR / "scaler_glcm.joblib")
    cache = load_cache() if (C.OUT_DIR / "preprocessed_224.npz").exists() else {}
    d = {}
    for s in ("train", "val", "test"):
        d[f"{s}_ak"], d[f"{s}_y"] = muat_aksara(s, cache, idx)
        d[f"{s}_co"] = muat_coretan(s)
    if a.smoke:
        for k in d:
            d[k] = d[k][:64]
        P.update(epochs_tahap1=1, epochs_tahap2=1)

    train = OEDataset(d["train_ak"], d["train_y"], d["train_co"], len(classes), scaler,
                      augment=True, shuffle=True)
    val = OEDataset(d["val_ak"], d["val_y"], d["val_co"], len(classes), scaler)

    model = keras.models.load_model(src, compile=False)
    base = next(l for l in model.layers if isinstance(l, keras.Model))
    wpath = str(OUT / "best.weights.h5")
    ckpt = keras.callbacks.ModelCheckpoint(wpath, monitor="val_loss", save_best_only=True,
                                           save_weights_only=True)

    def latih(lr, epochs):
        model.compile(keras.optimizers.Adam(lr), loss="categorical_crossentropy",
                      metrics=[AkurasiAksara()])
        es = keras.callbacks.EarlyStopping("val_loss", patience=P["patience"], restore_best_weights=True)
        return model.fit(train, validation_data=val, epochs=epochs, callbacks=[es, ckpt])

    t0 = time.time()
    print("== M2-OE | Tahap 1: kepala klasifikasi dilatih dengan coretan (backbone dibekukan) ==")
    base.trainable = False
    h1 = latih(P["lr_tahap1"], P["epochs_tahap1"])
    print(f"== Tahap 2: fine-tuning {P['unfreeze_last']} lapisan akhir backbone ==")
    base.trainable = True
    for layer in base.layers[:-P["unfreeze_last"]]:
        layer.trainable = False
    h2 = latih(P["lr_tahap2"], P["epochs_tahap2"])
    menit = (time.time() - t0) / 60

    model.load_weights(wpath)
    model.compile(loss="categorical_crossentropy")      # simpan tanpa metrik khusus agar mudah dimuat
    model.save(OUT / "best.keras")
    (OUT / "best.weights.h5").unlink(missing_ok=True)

    hist = pd.concat([pd.DataFrame(h1.history).assign(tahap=1),
                      pd.DataFrame(h2.history).assign(tahap=2)], ignore_index=True)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.to_csv(OUT / "history.csv", index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m, lab in [(axs[0], "loss", "Loss (aksara + coretan)"), (axs[1], "akurasi_aksara", "Akurasi aksara")]:
        ax.plot(hist.epoch, hist[m], label="train"); ax.plot(hist.epoch, hist[f"val_{m}"], label="validasi")
        ax.axvline(len(h1.history["loss"]) + .5, color="gray", ls="--", lw=1)
        ax.set(xlabel="Epoch", title=lab); ax.grid(alpha=.3); ax.legend()
    fig.suptitle("Kurva pelatihan M2-OE"); fig.tight_layout(); fig.savefig(OUT / "kurva_pelatihan.png", dpi=130)
    plt.close(fig)

    print("== Evaluasi: M2 (tanpa OE) vs M2-OE ==")
    hasil = [nilai("M2", keras.models.load_model(src, compile=False), d, scaler),
             nilai("M2-OE", model, d, scaler)]
    tabel = pd.DataFrame([h[0] for h in hasil])
    tabel.to_csv(OUT / "tabel_oe.csv", index=False)
    print(tabel.T.to_string(header=False))
    grafik(hasil, OUT / "grafik_oe.png")
    contoh_coretan(model, d["test_co"], scaler, classes, OUT / "contoh_coretan_oe.png")

    r = hasil[1][0]
    json.dump({"ambang": r["ambang_probabilitas"], "persentil_validasi_aksara": P["persentil_ambang"]},
              open(OUT / "ambang_ood.json", "w"), indent=1)
    json.dump({"model": "M2OE", "nama": "ResNet50-GLCM + Outlier Exposure", "arsitektur": ARCH,
               "glcm": GLCM, "dilanjutkan_dari": ASAL,
               "akurasi_uji": r["akurasi_uji_aksara"], "f1_macro": r["f1_macro_uji_aksara"],
               "jumlah_coretan": {s: len(d[f"{s}_co"]) for s in ("train", "val", "test")},
               "evaluasi_ood": r, "epoch_total": int(len(hist)),
               "epoch_tahap1": len(h1.history["loss"]), "waktu_latih_menit": round(menit, 1),
               "seed": C.SEED, "parameter": P},
              open(OUT / "ringkasan_hasil.json", "w"), indent=1)
    print(f"Selesai. Hasil di {OUT}")


if __name__ == "__main__":
    main()
