"""Jalankan sekali setelah EDA.py:  python run_preprocessing.py

Langkah:
 1. Baca split_train/val/test.csv dari EDA (70/15/15, stratified kelas+sumber, seed 42)
    dan ubah path Windows menjadi path relatif.
 2. Preprocessing semua citra -> grayscale 224x224.
 3. Audit hampir-duplikat (korelasi piksel 48x48 >= DUP_THRESHOLD). Dari tiap grup
    duplikat hanya satu citra dipertahankan (prioritas: train > val > test) agar
    tidak ada citra yang sama di data latih dan data uji.
 4. Ekstraksi 32 fitur GLCM (citra tanpa augmentasi); StandardScaler di-fit HANYA
    pada data latih.
 5. Simpan cache citra, split final, gambar contoh untuk laporan, dan ringkasan.
"""
import json
from itertools import combinations

import cv2
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse.csgraph import connected_components
from sklearn.preprocessing import StandardScaler

import config as C
from augmentation import augment_gray
from glcm_features import N_FEATURES, feature_names, glcm_features
from preprocessing import load_gray, preprocess_gray, to_local_path

C.OUT_DIR.mkdir(exist_ok=True)
SPLITS = ["train", "val", "test"]

# 1. Split dari EDA ----------------------------------------------------------
parts = []
for s in SPLITS:
    d = pd.read_csv(C.EDA_DIR / f"split_{s}.csv")
    d["abspath"] = d.path.map(to_local_path)
    d["relpath"] = d.abspath.map(lambda p: p.relative_to(C.DATASET_DIR).as_posix())
    d["split"] = s
    parts.append(d)
df = pd.concat(parts, ignore_index=True).drop(columns="path")
n_awal = df.split.value_counts().reindex(SPLITS)

missing = df[~df.abspath.map(lambda p: p.exists())]
if len(missing):
    print(f"PERINGATAN: {len(missing)} file di CSV tidak ditemukan, dilewati:")
    print(missing.relpath.to_string())
    df = df.drop(missing.index).reset_index(drop=True)

for a, b in combinations(SPLITS, 2):
    inter = set(df[df.split == a].relpath) & set(df[df.split == b].relpath)
    assert not inter, f"Kebocoran: {len(inter)} file ada di {a} dan {b}"

# 2. Preprocessing -----------------------------------------------------------
print(f"Preprocessing {len(df)} citra ke {C.IMG_SIZE}x{C.IMG_SIZE} ...")
imgs, infos = [], []
for i, p in enumerate(df.abspath):
    g, info = preprocess_gray(p, return_info=True)
    imgs.append(g); infos.append(info)
    if (i + 1) % 300 == 0:
        print(f"  {i + 1}/{len(df)}")
imgs = np.stack(imgs)
df = pd.concat([df, pd.DataFrame(infos)], axis=1)

# 3. Audit hampir-duplikat ---------------------------------------------------
v = np.stack([cv2.resize(g, (48, 48), interpolation=cv2.INTER_AREA).ravel() for g in imgs]).astype(np.float32)
v -= v.mean(1, keepdims=True); v /= np.linalg.norm(v, axis=1, keepdims=True) + 1e-8
S = v @ v.T
np.fill_diagonal(S, 0)
ii, jj = np.where(np.triu(S >= C.DUP_THRESHOLD, k=1))
pairs = pd.DataFrame({"a": df.relpath[ii].values, "split_a": df.split[ii].values,
                      "b": df.relpath[jj].values, "split_b": df.split[jj].values,
                      "korelasi": S[ii, jj].round(4)})
pairs["lintas_split"] = pairs.split_a != pairs.split_b
pairs["beda_label"] = df.label[ii].values != df.label[jj].values
pairs.to_csv(C.OUT_DIR / "pasangan_hampir_duplikat.csv", index=False)

adj = np.zeros_like(S, dtype=bool); adj[ii, jj] = adj[jj, ii] = True
_, comp = connected_components(adj, directed=False)
rank = df.split.map({s: k for k, s in enumerate(SPLITS)}).to_numpy()
drop = []
for c in np.unique(comp[ii]) if len(ii) else []:
    members = np.where(comp == c)[0]
    keep = members[np.lexsort((members, rank[members]))[0]]   # train dulu, lalu val, test
    drop += [m for m in members if m != keep]
removed = df.loc[drop, ["relpath", "label", "source", "split"]]
removed.to_csv(C.OUT_DIR / "citra_dibuang_duplikat.csv", index=False)

# Gambar bukti beberapa pasangan duplikat (untuk laporan)
if len(pairs):
    show = pairs.sort_values("lintas_split", ascending=False).head(6)
    idx = {r: k for k, r in enumerate(df.relpath)}
    fig, axs = plt.subplots(len(show), 2, figsize=(4.2, 2.1 * len(show)), squeeze=False)
    for r, (_, row) in enumerate(show.iterrows()):
        for c, (p, s) in enumerate([(row.a, row.split_a), (row.b, row.split_b)]):
            axs[r, c].imshow(load_gray(df.abspath[idx[p]]), cmap="gray", vmin=0, vmax=255)
            axs[r, c].set_title(f"{p} [{s}]", fontsize=6); axs[r, c].axis("off")
        axs[r, 0].text(1.05, -0.02, f"r = {row.korelasi:.3f}", transform=axs[r, 0].transAxes,
                       fontsize=7, ha="center")
    fig.suptitle("Contoh pasangan citra hampir-duplikat", fontsize=9); fig.tight_layout()
    fig.savefig(C.OUT_DIR / "08_contoh_duplikat.png", dpi=130); plt.close(fig)

keep_mask = ~df.index.isin(drop)
df, imgs = df[keep_mask].reset_index(drop=True), imgs[keep_mask]

# 4. GLCM + scaler -----------------------------------------------------------
print("Ekstraksi fitur GLCM ...")
F = np.stack([glcm_features(g) for g in imgs])
assert F.shape[1] == N_FEATURES == 32
scaler = StandardScaler().fit(F[df.split == "train"])
joblib.dump(scaler, C.OUT_DIR / "scaler_glcm.joblib")
glcm_df = pd.DataFrame(F, columns=feature_names())
glcm_df.insert(0, "relpath", df.relpath); glcm_df.insert(1, "label", df.label)
glcm_df.insert(2, "split", df.split)
glcm_df.to_csv(C.OUT_DIR / "glcm_features_raw.csv", index=False)

# 5. Simpan hasil ------------------------------------------------------------
class_names = sorted(df.label.unique())
json.dump(class_names, open(C.OUT_DIR / "class_names.json", "w"), indent=1)
for s in SPLITS:
    df[df.split == s][["relpath", "label", "source"]].to_csv(C.OUT_DIR / f"split_{s}.csv", index=False)
np.savez_compressed(C.OUT_DIR / "preprocessed_224.npz", images=imgs,
                    relpath=df.relpath.to_numpy().astype(str))
df.drop(columns="abspath").to_csv(C.OUT_DIR / "log_preprocessing.csv", index=False)

rng = np.random.default_rng(C.SEED)
pick = pd.concat([g.sample(3, random_state=C.SEED)
                  for _, g in df[df.split == "train"].groupby("source")])
fig, axs = plt.subplots(len(pick), 2, figsize=(5.2, 2.6 * len(pick)))
for r, (_, row) in enumerate(pick.iterrows()):
    axs[r, 0].imshow(load_gray(row.abspath), cmap="gray", vmin=0, vmax=255)
    axs[r, 0].set_title(f"Asli ({row.orig_w}x{row.orig_h}) - {row.label}/{row.source}", fontsize=7)
    axs[r, 1].imshow(imgs[row.name], cmap="gray", vmin=0, vmax=255)
    axs[r, 1].set_title(f"Crop+pad+resize {C.IMG_SIZE}x{C.IMG_SIZE}", fontsize=7)
for a in axs.ravel(): a.axis("off")
fig.suptitle("Contoh hasil preprocessing", fontsize=9); fig.tight_layout()
fig.savefig(C.OUT_DIR / "06_contoh_preprocessing.png", dpi=130); plt.close(fig)

n_aug = 5
pick = pick.iloc[[0, -1]]
fig, axs = plt.subplots(len(pick), n_aug + 1, figsize=(1.9 * (n_aug + 1), 2.1 * len(pick)))
for r, (_, row) in enumerate(pick.iterrows()):
    g = imgs[row.name]
    axs[r, 0].imshow(g, cmap="gray", vmin=0, vmax=255); axs[r, 0].set_title("Hasil preprocessing", fontsize=7)
    for k in range(1, n_aug + 1):
        axs[r, k].imshow(augment_gray(g, rng), cmap="gray", vmin=0, vmax=255)
        axs[r, k].set_title(f"Augmentasi {k}", fontsize=7)
for a in axs.ravel(): a.axis("off")
fig.suptitle("Contoh augmentasi data latih (rotasi, zoom, shift)", fontsize=9); fig.tight_layout()
fig.savefig(C.OUT_DIR / "07_contoh_augmentasi.png", dpi=130); plt.close(fig)

# Ringkasan ------------------------------------------------------------------
n_akhir = df.split.value_counts().reindex(SPLITS)
per_kelas = df[df.split == "train"].label.value_counts()
small = ((df.crop_h < C.IMG_SIZE) | (df.crop_w < C.IMG_SIZE)).sum()
summary = f"""RINGKASAN PREPROCESSING
- Split awal (EDA)         : {n_awal['train']:,} / {n_awal['val']:,} / {n_awal['test']:,} (train/val/test, stratified kelas+sumber)
- Pasangan hampir-duplikat : {len(pairs)} (r >= {C.DUP_THRESHOLD}; lintas split: {int(pairs.lintas_split.sum())}; beda label: {int(pairs.beda_label.sum())})
- Citra dibuang (duplikat) : {len(removed)} (train {int((removed.split=='train').sum())}, val {int((removed.split=='val').sum())}, test {int((removed.split=='test').sum())})
- Split akhir              : {n_akhir['train']:,} / {n_akhir['val']:,} / {n_akhir['test']:,} = {len(df):,} citra, {len(class_names)} kelas
- Citra latih per kelas    : min {per_kelas.min()}, maks {per_kelas.max()}
- Ukuran keluaran          : {C.IMG_SIZE}x{C.IMG_SIZE}; grayscale untuk GLCM, disalin ke 3 kanal untuk CNN
- Crop karakter berhasil   : {int(df.crop_ok.sum()):,} / {len(df):,} (margin {int(C.CROP_MARGIN*100)}%)
- Area karakter < {C.IMG_SIZE}px    : {int(small)} citra (di-upscale)
- Latar gelap dibalik      : {int((df.bg < 127).sum())}
- Normalisasi CNN          : ResNet50 = caffe (BGR - mean ImageNet); MobileNetV2 = skala [-1, 1]
- Fitur GLCM per citra     : {N_FEATURES} (levels={C.GLCM_LEVELS}, d={C.GLCM_DISTANCES}, sudut={C.GLCM_ANGLES_DEG})
- Scaler GLCM di-fit pada  : data latih saja ({n_akhir['train']:,} citra)
- Augmentasi (train saja)  : rotasi +-{C.AUG_ROTATION_DEG} deg, zoom {C.AUG_ZOOM}, shift +-{int(C.AUG_SHIFT*100)}%, noise std {C.AUG_NOISE_STD}, tanpa flip
"""
print(summary)
(C.OUT_DIR / "ringkasan_preprocessing.txt").write_text(summary, encoding="utf-8")
