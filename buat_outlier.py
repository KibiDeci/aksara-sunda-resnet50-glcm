"""Membuat contoh citra NON-aksara (outlier) untuk Outlier Exposure.

    python buat_outlier.py            # hasil di folder outlier/{train,val,test}/

Jenis citra: coretan acak, spiral, bentuk geometris, gelombang/zigzag, arsiran,
kata dan angka Latin, serta campurannya. Sebagian dibuat bergaya foto kamera
(latar abu-abu, noise, blur) agar mirip variasi dataset.

Citra yang terlalu mirip aksara asli (korelasi piksel >= MAX_KORELASI terhadap
salah satu citra dataset setelah preprocessing) dibuang, agar model tidak
diajari bahwa bentuk aksara adalah "bukan aksara".

Data train/val/test memakai seed berbeda dan dibuat terpisah.
"""
import math
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

import config as C
from preprocessing import preprocess_gray

OUT = C.BASE_DIR / "outlier"
JUMLAH = {"train": 1500, "val": 300, "test": 300}
SEED = {"train": 101, "val": 202, "test": 303}
MAX_KORELASI = 0.60
SIZE = 400
import matplotlib                                   # font DejaVu ikut terpasang bersama matplotlib
_MPL = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
FONTS = [str(p) for p in [_MPL / n for n in ("DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSerif.ttf",
                                             "DejaVuSans-Oblique.ttf", "DejaVuSansMono.ttf",
                                             "DejaVuSerif-Italic.ttf")] if p.exists()]
HURUF = "abcdefghkmnopqrstuvwxyABCDEFHKMPQRSTVXY"
ANGKA = "04689"


def _spline(rng, n, spread=0.35):
    c = SIZE / 2
    pts = np.array([[c + rng.uniform(-spread, spread) * SIZE, c + rng.uniform(-spread, spread) * SIZE]
                    for _ in range(n)])
    t = np.linspace(0, 1, n); tt = np.linspace(0, 1, n * 25)
    return np.stack([np.interp(tt, t, pts[:, 0]), np.interp(tt, t, pts[:, 1])], 1)


def _smooth(path, k=9):
    if len(path) < k: return path
    ker = np.ones(k) / k
    return np.stack([np.convolve(path[:, i], ker, mode="valid") for i in range(2)], 1)


def coretan(d, rng, w):
    d.line([tuple(p) for p in _smooth(_spline(rng, rng.randint(5, 12)), k=45)], fill=0, width=w, joint="curve")


def ikal(d, rng, w):
    """Coretan berikal seperti tulisan sambung (trokoid)."""
    n = rng.randint(2, 6); t = np.linspace(0, 2 * math.pi * n, 500)
    r = rng.uniform(20, 55); a = rng.uniform(.8, 1.6) * r
    x = t / t.max() * rng.uniform(150, 300) + a * np.sin(t)
    y = r * np.cos(t) * rng.uniform(.8, 2.2) + rng.uniform(-40, 40) * np.sin(t / n)
    x += SIZE / 2 - (x.min() + x.max()) / 2; y += SIZE / 2 - (y.min() + y.max()) / 2
    if rng.random() < .5:
        x, y = y - y.mean() + SIZE / 2, x - x.mean() + SIZE / 2      # versi tegak
    d.line(list(zip(x, y)), fill=0, width=w, joint="curve")


def spiral(d, rng, w):
    c = SIZE / 2 + rng.uniform(-30, 30); turns = rng.uniform(1.5, 4); r0 = rng.uniform(60, 150)
    th = np.linspace(0, 2 * math.pi * turns, 400); r = r0 * th / th.max()
    d.line([(c + x, c + y) for x, y in zip(r * np.cos(th), r * np.sin(th))], fill=0, width=w, joint="curve")


def bentuk(d, rng, w):
    c = SIZE / 2; s = rng.uniform(70, 150); k = rng.choice(["lingkaran", "kotak", "segitiga", "bintang", "elips2"])
    if k == "lingkaran":
        d.ellipse([c - s, c - s, c + s, c + s], outline=0, width=w)
    elif k == "kotak":
        d.rectangle([c - s, c - s * .7, c + s, c + s * .7], outline=0, width=w)
    elif k == "segitiga":
        d.polygon([(c, c - s), (c - s, c + s * .8), (c + s, c + s * .8)], outline=0, width=w)
    elif k == "bintang":
        pts = [(c + (s if i % 2 == 0 else s * .45) * math.cos(i * math.pi / 5 - math.pi / 2),
                c + (s if i % 2 == 0 else s * .45) * math.sin(i * math.pi / 5 - math.pi / 2)) for i in range(10)]
        d.polygon(pts, outline=0, width=w)
    else:
        d.ellipse([c - s, c - s * .5, c + s, c + s * .5], outline=0, width=w)
        d.ellipse([c - s * .5, c - s, c + s * .5, c + s], outline=0, width=w)


def gelombang(d, rng, w):
    x = np.linspace(40, SIZE - 40, 300); f = rng.uniform(2, 6); a = rng.uniform(30, 90)
    if rng.random() < .5:
        y = SIZE / 2 + a * np.sin(x / SIZE * f * 2 * math.pi)
    else:
        y = SIZE / 2 + a * (2 * np.abs((x / SIZE * f) % 1 - .5) * 2 - 1)
    for k in range(rng.choice([1, 1, 2])):
        d.line(list(zip(x, y + k * rng.uniform(40, 80))), fill=0, width=w, joint="curve")


def arsiran(d, rng, w):
    n = rng.randint(6, 14); ang = rng.uniform(0, math.pi)
    for i in range(n):
        o = (i - n / 2) * rng.uniform(15, 25); dx, dy = math.cos(ang), math.sin(ang)
        cx, cy = SIZE / 2 - dy * o, SIZE / 2 + dx * o
        d.line([(cx - dx * 140, cy - dy * 140), (cx + dx * 140, cy + dy * 140)], fill=0, width=max(2, w // 2))


def teks(d, rng, w):
    if not FONTS:
        return coretan(d, rng, w)
    s = "".join(rng.choice(HURUF) for _ in range(rng.randint(2, 5))) if rng.random() < .75 \
        else "".join(rng.choice(ANGKA) for _ in range(rng.randint(2, 4)))
    size = rng.randint(90, 160)
    f = ImageFont.truetype(rng.choice(FONTS), size)
    bb = d.textbbox((0, 0), s, font=f)
    while bb[2] - bb[0] > SIZE * .9 and size > 40:            # pastikan teks muat
        size -= 10; f = ImageFont.truetype(f.path, size); bb = d.textbbox((0, 0), s, font=f)
    d.text(((SIZE - (bb[2] - bb[0])) / 2, (SIZE - (bb[3] - bb[1])) / 2 - bb[1]), s, font=f, fill=0,
           stroke_width=max(0, w // 4), stroke_fill=0)


JENIS = [(coretan, .22), (ikal, .18), (spiral, .08), (bentuk, .13), (gelombang, .09), (arsiran, .05), (teks, .25)]


def buat_satu(rng):
    img = Image.new("L", (SIZE, SIZE), 255); d = ImageDraw.Draw(img)
    w = rng.choice([3, 4, 5, 6, 8, 10, 12, 14])
    fn = rng.choices([j for j, _ in JENIS], weights=[p for _, p in JENIS])[0]
    fn(d, rng, w)
    if rng.random() < .2:                                    # campuran dua jenis
        rng.choices([j for j, _ in JENIS], weights=[p for _, p in JENIS])[0](d, rng, w)
    a = np.asarray(img.rotate(rng.uniform(-25, 25), fillcolor=255, resample=Image.BILINEAR), np.float32)
    if rng.random() < .4:
        # Gaya foto kamera: goresan sedikit kabur dan tinta tidak pekat. Latar tetap putih
        # dan tinta tetap gelap (0-15) seperti data aksara setelah preprocessing, agar model
        # tidak belajar jalan pintas "latar abu-abu = bukan aksara".
        ink = rng.uniform(0, 15)
        a = ink + (a / 255) * (255 - ink)
        a = cv2.GaussianBlur(a, (0, 0), rng.uniform(.6, 1.6))
    return np.clip(a, 0, 255).astype(np.uint8)


def vektor(g):
    v = cv2.resize(g, (48, 48), interpolation=cv2.INTER_AREA).ravel().astype(np.float32)
    v -= v.mean(); return v / (np.linalg.norm(v) + 1e-8)


def main():
    print("Memuat referensi aksara untuk penyaringan kemiripan ...")
    ref = []
    for s in ("train", "val", "test"):
        df = pd.read_csv(C.OUT_DIR / f"split_{s}.csv")
        ref += [vektor(preprocess_gray(C.DATASET_DIR / r)) for r in df.relpath]
    ref = np.stack(ref)
    tmp = OUT / "_tmp.png"; OUT.mkdir(exist_ok=True)
    for split, n in JUMLAH.items():
        d = OUT / split; d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.png"): f.unlink()
        rng = random.Random(SEED[split]); k = dibuang = 0
        while k < n:
            a = buat_satu(rng)
            cv2.imwrite(str(tmp), a)
            g = preprocess_gray(tmp)
            if float((ref @ vektor(g)).max()) >= MAX_KORELASI:
                dibuang += 1; continue
            cv2.imwrite(str(d / f"outlier_{split}_{k:04d}.png"), a); k += 1
        print(f"{split}: {n} citra (dibuang karena mirip aksara: {dibuang})")
    tmp.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
