"""Evaluasi & perbandingan model yang sudah dilatih (M1-M4) pada data uji internal.

    python evaluasi.py

Membaca setiap hasil_Mx/best.keras yang ada, memprediksi data uji yang SAMA,
lalu menyimpan ke folder evaluasi_output/:
  prediksi_uji_Mx.csv          prediksi per citra (untuk analisis kesalahan)
  tabel_perbandingan.csv       akurasi (+ IK 95% Wilson), precision/recall/F1 makro, waktu, parameter
  uji_mcnemar.csv              M2 vs M1, M4 vs M3, M3 vs M1 (uji binomial eksak pada pasangan berbeda)
  f1_per_kelas.csv             F1 tiap kelas untuk tiap model
  grafik_perbandingan.png      akurasi & F1 makro dengan IK 95%
  salah_Mx.png                 galeri citra uji yang salah diprediksi
  ringkasan_evaluasi.txt       ringkasan teks untuk laporan
"""
import json
import math

import joblib
import keras
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import f1_score, precision_recall_fscore_support

import config as C
from data_loader import AksaraDataset, load_cache, load_split

MODELS = {"M1": ("resnet50", False, "ResNet50"), "M2": ("resnet50", True, "ResNet50-GLCM"),
          "M3": ("mobilenetv2", False, "MobileNetV2"), "M4": ("mobilenetv2", True, "MobileNetV2-GLCM")}
PAIRS = [("M2", "M1", "kontribusi GLCM pada ResNet50 (analisis utama)"),
         ("M4", "M3", "kontribusi GLCM pada MobileNetV2"),
         ("M1", "M3", "ResNet50 vs MobileNetV2")]
OUT = C.BASE_DIR / "evaluasi_output"


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar(a_ok, b_ok):
    """Uji McNemar eksak: hanya citra yang hasilnya berbeda antara dua model."""
    b = int(np.sum(a_ok & ~b_ok))     # A benar, B salah
    c = int(np.sum(~a_ok & b_ok))     # A salah, B benar
    p = binomtest(b, b + c, 0.5).pvalue if b + c else 1.0
    return b, c, p


def main():
    OUT.mkdir(exist_ok=True)
    classes = json.load(open(C.OUT_DIR / "class_names.json"))
    cache = load_cache()
    scaler = joblib.load(C.OUT_DIR / "scaler_glcm.joblib")
    test_df = load_split("test")
    labels = list(range(len(classes)))

    preds, rows, f1s = {}, [], {}
    for m, (arch, glcm, nama) in MODELS.items():
        ckpt = C.BASE_DIR / f"hasil_{m}" / "best.keras"
        if not ckpt.exists():
            print(f"{m}: belum dilatih, dilewati.")
            continue
        print(f"Mengevaluasi {m} ({nama}) ...")
        ds = AksaraDataset(test_df, classes, cache, arch=arch, with_glcm=glcm, scaler=scaler)
        prob = keras.models.load_model(ckpt).predict(ds, verbose=0)
        y, yhat = ds.y, prob.argmax(1)
        preds[m] = (y, yhat)

        pd.DataFrame({"relpath": test_df.relpath, "label": test_df.label, "source": test_df.source,
                      "prediksi": [classes[i] for i in yhat],
                      "probabilitas": prob.max(1).round(4), "benar": y == yhat}
                     ).to_csv(OUT / f"prediksi_uji_{m}.csv", index=False)

        k, n = int((y == yhat).sum()), len(y)
        lo, hi = wilson(k, n)
        p, r, f, _ = precision_recall_fscore_support(y, yhat, labels=labels, average="macro", zero_division=0)
        ring = C.BASE_DIR / f"hasil_{m}" / "ringkasan_hasil.json"
        rj = json.load(open(ring)) if ring.exists() else {}
        kamera = (test_df.source == "kamera").to_numpy()
        rows.append({"model": m, "nama": nama, "benar": k, "salah": n - k, "n_uji": n,
                     "akurasi": k / n, "ik95_bawah": lo, "ik95_atas": hi,
                     "precision_makro": p, "recall_makro": r, "f1_makro": f,
                     "akurasi_foto_kamera": float((y == yhat)[kamera].mean()),
                     "akurasi_sumber_lain": float((y == yhat)[~kamera].mean()),
                     "epoch_terbaik": rj.get("epoch_terbaik"), "waktu_latih_menit": rj.get("waktu_latih_menit"),
                     "inferensi_ms_per_citra": rj.get("waktu_inferensi_ms_per_citra"),
                     "jumlah_parameter": rj.get("jumlah_parameter")})
        f1s[m] = f1_score(y, yhat, labels=labels, average=None, zero_division=0)

        # galeri kesalahan
        wrong = np.where(y != yhat)[0][:36]          # maks. 36 contoh
        if len(wrong):
            cols = min(6, len(wrong)); rws = math.ceil(len(wrong) / cols)
            fig, axs = plt.subplots(rws, cols, figsize=(1.9 * cols, 2.2 * rws), squeeze=False)
            for ax in axs.ravel(): ax.axis("off")
            for ax, i in zip(axs.ravel(), wrong):
                ax.imshow(cache[test_df.relpath[i]], cmap="gray", vmin=0, vmax=255)
                ax.set_title(f"asli: {classes[y[i]]}\nprediksi: {classes[yhat[i]]}", fontsize=8)
            fig.suptitle(f"Citra uji yang salah diprediksi - {m} ({nama})", fontsize=9)
            fig.tight_layout(); fig.savefig(OUT / f"salah_{m}.png", dpi=130); plt.close(fig)

    if not rows:
        raise SystemExit("Belum ada model yang dilatih (hasil_Mx/best.keras).")

    tab = pd.DataFrame(rows)
    tab.round(4).to_csv(OUT / "tabel_perbandingan.csv", index=False)
    pd.DataFrame(f1s, index=classes).round(4).to_csv(OUT / "f1_per_kelas.csv")

    mc = []
    for a, b, ket in PAIRS:
        if a in preds and b in preds:
            (y, pa), (_, pb) = preds[a], preds[b]
            n10, n01, p = mcnemar(pa == y, pb == y)
            mc.append({"perbandingan": f"{a} vs {b}", "keterangan": ket,
                       f"{a}_benar_{b}_salah": n10, f"{a}_salah_{b}_benar": n01,
                       "p_value": round(p, 4), "signifikan_0.05": p < 0.05})
    mc = pd.DataFrame(mc)
    if len(mc):
        mc.to_csv(OUT / "uji_mcnemar.csv", index=False)

    # grafik
    fig, ax = plt.subplots(figsize=(1.6 + 1.5 * len(tab), 4))
    x = np.arange(len(tab)); w = .38
    ax.bar(x - w / 2, tab.akurasi * 100, w, label="Akurasi", color="#1f5f5b",
           yerr=[(tab.akurasi - tab.ik95_bawah) * 100, (tab.ik95_atas - tab.akurasi) * 100], capsize=4)
    ax.bar(x + w / 2, tab.f1_makro * 100, w, label="F1 makro", color="#8fb8b3")
    for i, r in tab.iterrows():
        ax.text(i - w / 2, r.akurasi * 100 + .3, f"{r.akurasi*100:.2f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + w / 2, r.f1_makro * 100 + .3, f"{r.f1_makro*100:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [f"{r.model}\n{r.nama}" for _, r in tab.iterrows()], fontsize=8)
    lo = max(0, math.floor(min(tab.ik95_bawah.min(), tab.f1_makro.min()) * 100) - 3)
    ax.set_ylim(lo, 101); ax.set_ylabel("%"); ax.grid(axis="y", alpha=.3); ax.legend(loc="lower right")
    ax.set_title("Kinerja pada data uji internal (garis = IK 95% Wilson untuk akurasi)", fontsize=9)
    fig.tight_layout(); fig.savefig(OUT / "grafik_perbandingan.png", dpi=130); plt.close(fig)

    # ringkasan teks
    lines = ["RINGKASAN EVALUASI (data uji internal)", ""]
    for _, r in tab.iterrows():
        lines.append(f"{r.model} {r.nama:17s}: akurasi {r.akurasi*100:.2f}% (IK95 {r.ik95_bawah*100:.1f}-"
                     f"{r.ik95_atas*100:.1f}%), F1 makro {r.f1_makro*100:.2f}%, salah {r.salah}/{r.n_uji}, "
                     f"kamera {r.akurasi_foto_kamera*100:.1f}% / lainnya {r.akurasi_sumber_lain*100:.1f}%")
    if len(mc):
        lines += ["", "Uji McNemar (eksak):"]
        for _, r in mc.iterrows():
            vals = list(r.values)
            lines.append(f"  {r.perbandingan} ({r.keterangan}): {vals[2]} vs {vals[3]} citra berbeda, "
                         f"p = {r.p_value:.4f} -> {'berbeda signifikan' if r['signifikan_0.05'] else 'tidak berbeda signifikan'}")
    txt = "\n".join(lines)
    print("\n" + txt)
    (OUT / "ringkasan_evaluasi.txt").write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
