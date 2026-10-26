# %% Konfigurasi
import os, hashlib, random, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image, ImageStat, UnidentifiedImageError

BASE_DIR    = Path(__file__).resolve().parent          # folder tempat EDA.py berada
DATASET_DIR = BASE_DIR / "dataset_gabung_fix"           # dataset harus sejajar dengan EDA.py
OUT_DIR     = BASE_DIR / "eda_output"; OUT_DIR.mkdir(exist_ok=True)
EXTS        = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
TARGET_SIZE = 224          # input ResNet50 & MobileNetV2 = 224x224
SEED        = 42
random.seed(SEED); np.random.seed(SEED)
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": .3})

# 1. Inventaris file
rows = []
for cls_dir in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
    for f in cls_dir.rglob("*"):
        if f.suffix.lower() in EXTS:
            rows.append({"path": str(f), "label": cls_dir.name})
df = pd.DataFrame(rows)
assert len(df), f"Tidak ada gambar ditemukan di {DATASET_DIR.resolve()}"
print(f"Total gambar : {len(df):,}")
print(f"Jumlah kelas : {df.label.nunique()}")

# 2. Properti tiap gambar (ukuran, mode, brightness, blur, hash, korup)
def laplacian_var(gray):
    """Ukuran ketajaman: variansi Laplacian (kecil = blur)."""
    g = gray.astype(np.float32)
    lap = (-4*g[1:-1,1:-1] + g[:-2,1:-1] + g[2:,1:-1] + g[1:-1,:-2] + g[1:-1,2:])
    return float(lap.var())

props = []
for p in df.path:
    d = {"corrupt": False}
    try:
        with open(p, "rb") as fh:
            d["md5"] = hashlib.md5(fh.read()).hexdigest()
        with Image.open(p) as im:
            im.load()
            d.update(width=im.width, height=im.height, mode=im.mode,
                     format=im.format, file_kb=os.path.getsize(p)/1024)
            gray = np.asarray(im.convert("L"))
            d.update(brightness=gray.mean(), contrast=gray.std(),
                     sharpness=laplacian_var(gray),
                     # latar terang/gelap: rata-rata piksel di tepi gambar
                     border_mean=np.concatenate([gray[0], gray[-1], gray[:,0], gray[:,-1]]).mean())
    except (UnidentifiedImageError, OSError, ValueError) as e:
        d["corrupt"] = True; d["error"] = str(e)
    props.append(d)
df = pd.concat([df, pd.DataFrame(props)], axis=1)
df["aspect"] = df.width / df.height
# Sumber gambar dari pola nama file (sesuaikan bila dataset lain): foto kamera HP = "IMG_..."
df["source"] = df.path.map(lambda p: "kamera" if Path(p).name.startswith("IMG_") else "lainnya")
df.to_csv(OUT_DIR / "metadata_raw.csv", index=False)

# 3. Ringkasan kualitas data
print("\n=== Kualitas data ===")
print("Gambar korup        :", int(df.corrupt.sum()))
dup = df[~df.corrupt & df.duplicated("md5", keep=False)]
print("Gambar duplikat     :", len(dup), f"({dup.md5.nunique()} grup)")
# duplikat lintas kelas = label bermasalah
cross = dup.groupby("md5").label.nunique()
print("Duplikat beda label :", int((cross > 1).sum()), "grup  <-- cek manual!")
print("\nMode warna:\n", df["mode"].value_counts(dropna=False).to_string())
print("\nFormat file:\n", df["format"].value_counts(dropna=False).to_string())
ok = df[~df.corrupt].copy()
print("\nUkuran gambar (px):")
print(ok[["width","height","aspect","file_kb"]].describe().round(1).to_string())
small = ok[(ok.width < TARGET_SIZE) | (ok.height < TARGET_SIZE)]
print(f"\nGambar lebih kecil dari {TARGET_SIZE}px (akan di-upscale): {len(small)} ({len(small)/len(ok):.1%})")

# 3b. Perbandingan antar sumber gambar (cek domain gap)
print("\n=== Per sumber gambar ===")
print(ok.groupby("source").agg(n=("path","size"), lebar_median=("width","median"), tinggi_median=("height","median"),
      brightness=("brightness","mean"), ketajaman_median=("sharpness","median")).round(1).to_string())
print("\nMode warna per sumber:\n", pd.crosstab(ok.source, ok["mode"]).to_string())
print("\nSumber per kelas (min/max):", ok.groupby(["label","source"]).size().unstack().agg(["min","max"]).to_dict())

# 4. Distribusi kelas
counts = ok.label.value_counts().sort_index()
ratio = counts.max() / counts.min()
print(f"\n=== Distribusi kelas ===\nMin: {counts.min()} ({counts.idxmin()}) | Max: {counts.max()} ({counts.idxmax()}) "
      f"| Rata-rata: {counts.mean():.1f} | Imbalance ratio: {ratio:.2f}x")
if ratio > 1.5: print("-> Kelas tidak seimbang: pertimbangkan augmentasi / class weight / stratified split.")

fig, ax = plt.subplots(figsize=(max(8, len(counts)*.35), 4.5))
ax.bar(counts.index, counts.values, color="#4472C4")
ax.axhline(counts.mean(), color="#C00000", ls="--", label=f"rata-rata = {counts.mean():.0f}")
ax.set(title="Distribusi Jumlah Gambar per Kelas", xlabel="Kelas", ylabel="Jumlah gambar")
ax.tick_params(axis="x", rotation=90); ax.legend()
fig.tight_layout(); fig.savefig(OUT_DIR / "01_distribusi_kelas.png"); plt.show()

# 5. Distribusi ukuran, aspect ratio, brightness, ketajaman
fig, axs = plt.subplots(2, 3, figsize=(14, 7))
for ax, (col, t) in zip(axs.ravel(), [("width","Lebar (px)"),("height","Tinggi (px)"),("aspect","Aspect ratio (w/h)"),
                                       ("brightness","Brightness rata-rata"),("contrast","Kontras (std piksel)"),
                                       ("sharpness","Ketajaman (var. Laplacian)")]):
    ax.hist(ok[col], bins=40, color="#4472C4", edgecolor="white"); ax.set_title(t)
fig.suptitle("Properti Citra", y=1.0); fig.tight_layout()
fig.savefig(OUT_DIR / "02_properti_citra.png"); plt.show()

# gambar yang mencurigakan (outlier IQR pada ketajaman & brightness)
def iqr_out(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return (s < q1 - 1.5*i) | (s > q3 + 1.5*i)
susp = ok[iqr_out(ok.sharpness) | iqr_out(ok.brightness)]
susp[["path","label","brightness","sharpness"]].to_csv(OUT_DIR / "gambar_outlier.csv", index=False)
print(f"Outlier brightness/ketajaman: {len(susp)} gambar -> eda_output/gambar_outlier.csv")

# 6. Sampel gambar per kelas
def show_samples(n_cls=None, per_cls=5, fname="03_sampel_per_kelas.png"):
    classes = sorted(ok.label.unique())[:n_cls]
    fig, axs = plt.subplots(len(classes), per_cls, figsize=(per_cls*1.6, len(classes)*1.6), squeeze=False)
    for i, c in enumerate(classes):
        sub = ok[ok.label == c]
        pick = sub.sample(min(per_cls, len(sub)), random_state=SEED).path.tolist()
        for j in range(per_cls):
            ax = axs[i][j]; ax.axis("off")
            if j < len(pick): ax.imshow(Image.open(pick[j]).convert("RGB"))
            if j == 0: ax.set_title(c, fontsize=8, loc="left")
    fig.tight_layout(); fig.savefig(OUT_DIR / fname); plt.show()
show_samples(n_cls=12)          # ubah n_cls=None untuk menampilkan semua kelas

# 7. Rata-rata citra per kelas (melihat pola umum bentuk aksara)
def mean_images(size=64, max_cls=24):
    classes = sorted(ok.label.unique())[:max_cls]
    cols = 6; rws = int(np.ceil(len(classes)/cols))
    fig, axs = plt.subplots(rws, cols, figsize=(cols*1.8, rws*1.9), squeeze=False)
    for ax in axs.ravel(): ax.axis("off")
    for ax, c in zip(axs.ravel(), classes):
        paths = ok[ok.label == c].path.tolist()[:100]
        arr = np.mean([np.asarray(Image.open(p).convert("L").resize((size, size))) for p in paths], axis=0)
        ax.imshow(arr, cmap="gray"); ax.set_title(c, fontsize=8)
    fig.suptitle("Rata-rata citra per kelas"); fig.tight_layout()
    fig.savefig(OUT_DIR / "04_rata_rata_citra.png"); plt.show()
mean_images()

# 8. Brightness & latar per kelas (deteksi bias, mis. kelas tertentu selalu lebih gelap)
fig, ax = plt.subplots(figsize=(max(8, ok.label.nunique()*.35), 4.5))
order = sorted(ok.label.unique())
ax.boxplot([ok[ok.label == c].brightness for c in order], showfliers=False)
ax.set_xticks(range(1, len(order)+1)); ax.set_xticklabels(order)
ax.set(title="Brightness per Kelas", ylabel="Brightness"); ax.tick_params(axis="x", rotation=90)
fig.tight_layout(); fig.savefig(OUT_DIR / "05_brightness_per_kelas.png"); plt.show()
dark_bg = (ok.border_mean < 127).mean()
print(f"Latar gelap (tepi < 127): {dark_bg:.1%} | latar terang: {1-dark_bg:.1%}")

# 9. Bersihkan data (hapus korup & duplikat persis) + split stratified (kelas + sumber) 70/15/15
from sklearn.model_selection import train_test_split
clean = ok.drop_duplicates("md5", keep="first").reset_index(drop=True)
clean["strata"] = clean.label + "_" + clean.source   # stratify per kelas DAN sumber
print(f"\nSetelah dibersihkan: {len(clean):,} gambar (dibuang {len(df)-len(clean)})")

too_few = clean.strata.value_counts()[lambda s: s < 3]
if len(too_few): print("PERINGATAN: kelas dengan <3 gambar tidak bisa di-split stratified:", list(too_few.index))

train, temp = train_test_split(clean, test_size=.30, stratify=clean.strata, random_state=SEED)
val, test   = train_test_split(temp,  test_size=.50, stratify=temp.strata,  random_state=SEED)
for name, part in [("train", train), ("val", val), ("test", test)]:
    part[["path","label","source"]].to_csv(OUT_DIR / f"split_{name}.csv", index=False)
    print(f"{name:5s}: {len(part):6,d} gambar")
clean[["path","label","source"]].to_csv(OUT_DIR / "dataset_bersih.csv", index=False)

# 10. Ringkasan untuk laporan (Dataset & Persiapan Data)
summary = f"""
RINGKASAN EDA
- Total gambar mentah      : {len(df):,}
- Jumlah kelas             : {df.label.nunique()}
- Gambar korup / duplikat  : {int(df.corrupt.sum())} / {len(dup)}
- Ukuran median (w x h)    : {int(ok.width.median())} x {int(ok.height.median())} px
- Imbalance ratio kelas    : {ratio:.2f}x (min {counts.min()}, max {counts.max()})
- Mode warna               : {", ".join(f"{k}={v}" for k,v in df["mode"].value_counts().items())} (konversi ke RGB)
- Sumber gambar            : {", ".join(f"{k}={v}" for k,v in ok.source.value_counts().items())}
- Gambar < {TARGET_SIZE}px           : {len(small)} ({len(small)/len(ok):.1%})
- Setelah pembersihan      : {len(clean):,} gambar
- Split train/val/test     : {len(train):,} / {len(val):,} / {len(test):,}
"""
print(summary); (OUT_DIR / "ringkasan_eda.txt").write_text(summary, encoding="utf-8")