"""Uji eksternal: menguji model final pada tulisan tangan penulis baru.

Susun data seperti ini (sejajar dengan config.py):

    uji_eksternal/
    ├── a/
    │   ├── budi_a.jpg
    │   ├── sari_a.jpg
    │   └── ...
    ├── ae/
    └── ... (30 folder kelas, nama sama dengan dataset_gabung_fix)

Nama berkas diawali nama/kode penulis lalu garis bawah (budi_..., p01_...),
agar akurasi per penulis bisa dihitung.

    python uji_eksternal.py              # menguji semua model yang punya hasil_Mx/best.keras
    python uji_eksternal.py --model M2   # hanya model tertentu

PENTING: data ini hanya untuk evaluasi akhir. Jangan dipakai untuk melatih,
memilih parameter, atau memilih model.

Keluaran di uji_eksternal_output/:
  tabel_uji_eksternal.csv     akurasi (+ IK 95%), precision, recall, macro-F1 per model
  akurasi_per_penulis.csv     akurasi tiap penulis per model
  prediksi_eksternal_Mx.csv   prediksi per citra
  salah_eksternal_Mx.png      galeri citra yang salah ditebak
  ringkasan_uji_eksternal.txt ringkasan untuk laporan
"""
import argparse
import json
import math
from pathlib import Path

import joblib
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

import config as C
from glcm_features import glcm_features
from preprocessing import gray_to_rgb, normalize_for, preprocess_gray

MODELS = {"M1": ("resnet50", False, "ResNet50"), "M2": ("resnet50", True, "ResNet50-GLCM"),
          "M3": ("mobilenetv2", False, "MobileNetV2"), "M4": ("mobilenetv2", True, "MobileNetV2-GLCM")}
DATA = C.BASE_DIR / "uji_eksternal"
OUT = C.BASE_DIR / "uji_eksternal_output"
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODELS, nargs="*")
    a = ap.parse_args()

    classes = json.load(open(C.OUT_DIR / "class_names.json"))
    if not DATA.exists():
        raise SystemExit(f"Folder {DATA} tidak ditemukan. Lihat petunjuk di bagian atas skrip.")
    unknown = [d.name for d in DATA.iterdir() if d.is_dir() and d.name not in classes]
    if unknown:
        raise SystemExit(f"Nama folder tidak dikenal: {unknown}. Harus sama dengan nama kelas: {classes}")

    rows = [{"path": p, "label": p.parent.name, "penulis": p.stem.split("_")[0]}
            for p in sorted(DATA.glob("*/*")) if p.suffix.lower() in EXTS]
    if not rows:
        raise SystemExit("Tidak ada citra di folder uji_eksternal.")
    df = pd.DataFrame(rows)
    print(f"Data eksternal: {len(df)} citra, {df.label.nunique()} kelas, {df.penulis.nunique()} penulis")
    kosong = sorted(set(classes) - set(df.label))
    if kosong:
        print(f"PERINGATAN: kelas tanpa data: {kosong}")

    print("Preprocessing ...")
    grays = np.stack([preprocess_gray(p) for p in df.path])
    feats = np.stack([glcm_features(g) for g in grays])
    scaler = joblib.load(C.OUT_DIR / "scaler_glcm.joblib")
    y = df.label.map({c: i for i, c in enumerate(classes)}).to_numpy()
    present = sorted(set(y))                     # macro-F1 hanya atas kelas yang ada di data
    OUT.mkdir(exist_ok=True)

    hasil, per_penulis = [], {}
    for m in (a.model or MODELS):
        arch, glcm, nama = MODELS[m]
        ckpt = C.BASE_DIR / f"hasil_{m}" / "best.keras"
        if not ckpt.exists():
            print(f"{m}: {ckpt} tidak ada, dilewati.")
            continue
        print(f"Menguji {m} ({nama}) ...")
        model = keras.models.load_model(ckpt)
        x = normalize_for(arch, np.stack([gray_to_rgb(g) for g in grays]))
        inp = [x, scaler.transform(feats).astype(np.float32)] if glcm else x
        prob = model.predict(inp, batch_size=32, verbose=0)
        yhat = prob.argmax(1)
        benar = yhat == y

        pd.DataFrame({"berkas": [str(p.relative_to(DATA)) for p in df.path], "label": df.label,
                      "penulis": df.penulis, "prediksi": [classes[i] for i in yhat],
                      "probabilitas": prob.max(1).round(4), "benar": benar}
                     ).to_csv(OUT / f"prediksi_eksternal_{m}.csv", index=False)

        k, n = int(benar.sum()), len(y)
        lo, hi = wilson(k, n)
        p, r, f, _ = precision_recall_fscore_support(y, yhat, labels=present, average="macro", zero_division=0)
        hasil.append({"model": m, "nama": nama, "n_citra": n, "benar": k, "akurasi": k / n,
                      "ik95_bawah": lo, "ik95_atas": hi, "precision_makro": p, "recall_makro": r, "f1_makro": f})
        per_penulis[m] = pd.Series(benar).groupby(df.penulis.values).mean()

        salah = np.where(~benar)[0][:36]
        if len(salah):
            cols = min(6, len(salah)); rws = math.ceil(len(salah) / cols)
            fig, axs = plt.subplots(rws, cols, figsize=(1.9 * cols, 2.3 * rws), squeeze=False)
            for ax in axs.ravel(): ax.axis("off")
            for ax, i in zip(axs.ravel(), salah):
                ax.imshow(grays[i], cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"{df.penulis[i]}\nasli {classes[y[i]]} → {classes[yhat[i]]}", fontsize=7)
            fig.suptitle(f"Kesalahan uji eksternal - {m} ({nama})", fontsize=9)
            fig.tight_layout(); fig.savefig(OUT / f"salah_eksternal_{m}.png", dpi=130); plt.close(fig)

    if not hasil:
        raise SystemExit("Tidak ada model yang bisa diuji.")
    tab = pd.DataFrame(hasil)
    tab.round(4).to_csv(OUT / "tabel_uji_eksternal.csv", index=False)
    pd.DataFrame(per_penulis).round(4).to_csv(OUT / "akurasi_per_penulis.csv")

    lines = [f"RINGKASAN UJI EKSTERNAL ({len(df)} citra, {df.penulis.nunique()} penulis, "
             f"{df.label.nunique()} kelas)", ""]
    for _, r in tab.iterrows():
        lines.append(f"{r.model} {r.nama:17s}: akurasi {r.akurasi*100:.2f}% "
                     f"(IK95 {r.ik95_bawah*100:.1f}-{r.ik95_atas*100:.1f}%), macro-F1 {r.f1_makro*100:.2f}%, "
                     f"benar {r.benar}/{r.n_citra}")
    lines += ["", "Akurasi per penulis:", pd.DataFrame(per_penulis).mul(100).round(1).to_string()]
    txt = "\n".join(lines)
    print("\n" + txt)
    (OUT / "ringkasan_uji_eksternal.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
