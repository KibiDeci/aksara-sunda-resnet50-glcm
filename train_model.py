"""Pelatihan M1-M4 (Bab 4 laporan). Jalankan SETELAH run_preprocessing.py.

    python train_model.py --model M1     # ResNet50          (baseline utama)
    python train_model.py --model M2     # ResNet50-GLCM     (model usulan)
    python train_model.py --model M3     # MobileNetV2       (baseline pembanding)
    python train_model.py --model M4     # MobileNetV2-GLCM

Dua tahap: (1) transfer learning, backbone dibekukan; (2) fine-tuning sebagian
lapisan akhir dengan learning rate lebih kecil. Early stopping + simpan bobot
terbaik berdasarkan val_loss. Semua hasil di folder hasil_<MODEL>/:
  best.keras, history.csv, kurva_pelatihan.png, confusion_matrix.png/.csv,
  classification_report.txt, ringkasan_hasil.json
"""
import argparse
import json
import time

import joblib
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

import config as C
from data_loader import AksaraDataset, load_cache, load_split

MODELS = {"M1": ("resnet50", False), "M2": ("resnet50", True),
          "M3": ("mobilenetv2", False), "M4": ("mobilenetv2", True)}
NAMA = {"M1": "ResNet50", "M2": "ResNet50-GLCM", "M3": "MobileNetV2", "M4": "MobileNetV2-GLCM"}

# Parameter awal (isi tabel "Parameter awal - Nilai usulan" di laporan).
# Nilai ini SAMA untuk M1-M4 agar perbandingan terkontrol.
P = dict(dense_units=256, dropout=0.5, lr_stage1=1e-3, lr_stage2=1e-5,
         epochs_stage1=30, epochs_stage2=30, unfreeze_last=30, patience=5,
         batch_size=C.BATCH_SIZE)


def build_model(arch, with_glcm, n_classes, weights="imagenet"):
    img_in = keras.Input((C.IMG_SIZE, C.IMG_SIZE, 3), name="citra")
    Base = keras.applications.ResNet50 if arch == "resnet50" else keras.applications.MobileNetV2
    base = Base(include_top=False, weights=weights,
                input_shape=(C.IMG_SIZE, C.IMG_SIZE, 3), pooling="avg")
    base.trainable = False
    x = base(img_in, training=False)          # BatchNorm tetap mode inferensi
    inputs = img_in
    if with_glcm:                             # feature-level fusion
        glcm_in = keras.Input((32,), name="glcm")
        x = layers.Concatenate()([x, glcm_in])
        inputs = [img_in, glcm_in]
    x = layers.Dense(P["dense_units"], activation="relu")(x)
    x = layers.Dropout(P["dropout"])(x)
    out = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, out), base


def plot_history(hist, n_stage1, path, title):
    ep = np.arange(1, len(hist) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for ax, m, lab in [(axs[0], "loss", "Loss"), (axs[1], "accuracy", "Akurasi")]:
        ax.plot(ep, hist[m], label="train"); ax.plot(ep, hist[f"val_{m}"], label="validasi")
        ax.axvline(n_stage1 + 0.5, color="gray", ls="--", lw=1)
        ax.text(n_stage1 + 0.7, ax.get_ylim()[1], "fine-tuning", va="top", fontsize=8, color="gray")
        ax.set(xlabel="Epoch", ylabel=lab, title=lab); ax.grid(alpha=.3); ax.legend()
    fig.suptitle(title); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_cm(cm, classes, path, title):
    cmn = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes, rotation=90, fontsize=8)
    ax.set_yticks(range(len(classes)), classes, fontsize=8)
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=6,
                        color="white" if cmn[i, j] > .5 else "black")
    ax.set(xlabel="Prediksi", ylabel="Label sebenarnya", title=title)
    fig.colorbar(im, fraction=.046); fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--smoke", action="store_true", help="uji cepat tanpa bobot ImageNet")
    a = ap.parse_args()
    arch, with_glcm = MODELS[a.model]
    keras.utils.set_random_seed(C.SEED)
    out = C.BASE_DIR / f"hasil_{a.model}"; out.mkdir(exist_ok=True)

    classes = json.load(open(C.OUT_DIR / "class_names.json"))
    kw = dict(class_names=classes, cache=load_cache(), arch=arch, with_glcm=with_glcm,
              scaler=joblib.load(C.OUT_DIR / "scaler_glcm.joblib"), batch_size=P["batch_size"])
    tr, va, te = (load_split(s) for s in ("train", "val", "test"))
    if a.smoke:
        tr, va, te = tr.head(64), va.head(32), te.head(32)
        P.update(epochs_stage1=2, epochs_stage2=2)
    train = AksaraDataset(tr, augment=True, shuffle=True, **kw)
    val, test = AksaraDataset(va, **kw), AksaraDataset(te, **kw)

    model, base = build_model(arch, with_glcm, len(classes), None if a.smoke else "imagenet")
    ckpt = str(out / "best.keras")
    checkpoint = keras.callbacks.ModelCheckpoint(ckpt, monitor="val_loss", save_best_only=True)

    def callbacks():
        return [keras.callbacks.EarlyStopping("val_loss", patience=P["patience"],
                                              restore_best_weights=True), checkpoint]

    t0 = time.time()
    print(f"== {a.model} {NAMA[a.model]} | Tahap 1: transfer learning (backbone dibekukan) ==")
    model.compile(keras.optimizers.Adam(P["lr_stage1"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train, validation_data=val, epochs=P["epochs_stage1"], callbacks=callbacks())

    print(f"== Tahap 2: fine-tuning {P['unfreeze_last']} lapisan akhir backbone ==")
    base.trainable = True
    for layer in base.layers[:-P["unfreeze_last"]]:
        layer.trainable = False
    model.compile(keras.optimizers.Adam(P["lr_stage2"]),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train, validation_data=val, epochs=P["epochs_stage2"], callbacks=callbacks())
    train_time = time.time() - t0

    hist = pd.concat([pd.DataFrame(h1.history).assign(tahap=1),
                      pd.DataFrame(h2.history).assign(tahap=2)], ignore_index=True)
    hist.insert(0, "epoch", np.arange(1, len(hist) + 1))
    hist.to_csv(out / "history.csv", index=False)
    plot_history(hist, len(h1.history["loss"]), out / "kurva_pelatihan.png",
                 f"Kurva pelatihan {a.model} ({NAMA[a.model]})")

    # Evaluasi pada data uji internal memakai bobot terbaik (val_loss terendah)
    model = keras.models.load_model(ckpt)
    y_true = test.y
    t1 = time.time()
    y_pred = np.argmax(model.predict(test, verbose=0), axis=1)
    infer_ms = (time.time() - t1) / len(y_true) * 1000
    labels = list(range(len(classes)))
    rep = classification_report(y_true, y_pred, labels=labels, target_names=classes,
                                digits=4, zero_division=0)
    print(rep)
    (out / "classification_report.txt").write_text(rep, encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=classes, columns=classes).to_csv(out / "confusion_matrix.csv")
    plot_cm(cm, classes, out / "confusion_matrix.png", f"Confusion matrix {a.model} - data uji")

    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                 average="macro", zero_division=0)
    pairs = [(classes[i], classes[j], int(cm[i, j])) for i, j in zip(*np.nonzero(cm))
             if i != j]
    best_ep = int(hist.val_loss.idxmin()) + 1
    summary = {
        "model": a.model, "nama": NAMA[a.model], "arsitektur": arch, "glcm": with_glcm,
        "akurasi_uji": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(p, 4), "recall_macro": round(r, 4), "f1_macro": round(f, 4),
        "jumlah_data_uji": int(len(y_true)),
        "epoch_total": int(len(hist)), "epoch_tahap1": len(h1.history["loss"]),
        "epoch_terbaik": best_ep, "val_loss_terbaik": round(float(hist.val_loss.min()), 4),
        "val_akurasi_pada_epoch_terbaik": round(float(hist.val_accuracy[best_ep - 1]), 4),
        "waktu_latih_menit": round(train_time / 60, 1),
        "waktu_inferensi_ms_per_citra": round(infer_ms, 2),
        "jumlah_parameter": int(model.count_params()),
        "pasangan_kelas_tertukar_terbanyak": sorted(pairs, key=lambda t: -t[2])[:5],
        "seed": C.SEED, "parameter": P,
    }
    json.dump(summary, open(out / "ringkasan_hasil.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "parameter"}, indent=1))


if __name__ == "__main__":
    main()
