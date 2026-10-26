# Tutorial Implementasi: Model ResNet50-GLCM dan Aplikasi

Dua bagian:
- **Bagian A**: melatih model (M1 baseline, M2 model usulan) di Google Colab.
- **Bagian B**: menjalankan aplikasi web di laptop memakai model hasil pelatihan.

Prasyarat: zip sudah berisi `dataset_gabung_fix\`, `eda_output\` (dari zip asli), dan semua file dari zip ini.

---

## Bagian A: Implementasi model (Google Colab)

Colab dipakai karena ResNet50 butuh GPU; di CPU laptop pelatihan bisa berjam-jam.

### A1. Upload zip ke Google Drive
Upload **Preprocessing_ResNet50_GLCM.zip** (satu file, tidak perlu diekstrak) ke **My Drive**, lalu upload juga **Pelatihan_M1_M2.ipynb** ke My Drive.

### A2. Buka notebook dan jalankan
1. Klik kanan `Pelatihan_M1_M2.ipynb` di Drive, lalu pilih **Open with → Google Colaboratory**.
2. **Runtime → Change runtime type → T4 GPU → Save**.
3. **Runtime → Run all**, lalu izinkan akses Google Drive.
4. Sel pertama mengekstrak zip ke `My Drive/Preprocessing_ResNet50_GLCM/` dan harus menampilkan `Folder kelas: 30 | eda_output ada: True`.
5. Sel kedua harus menampilkan `GPU: [PhysicalDevice(... 'GPU')]`. Kalau kosong, ulangi langkah 2 lalu Run all lagi.

### A3. Preprocessing
Jalankan sel "1. Preprocessing" (±1 menit). Ringkasan harus menampilkan:
```
Split akhir : 1,244 / 266 / 266 = 1,776 citra, 30 kelas
```
Yang terjadi: citra dipotong ke area karakter, diberi padding, di-resize 224×224, duplikat dibuang, dan 32 fitur GLCM dihitung. Hasilnya di `preprocessing_output/`.

### A4. Latih baseline M1 (ResNet50)
Sel M1 berjalan otomatis lewat Run all (±10–30 menit).
- **Tahap 1:** ResNet50 dibekukan, hanya lapisan klasifikasi yang belajar.
- **Tahap 2:** 30 lapisan terakhir ikut dilatih (fine-tuning) dengan learning rate kecil.
- Pelatihan berhenti otomatis kalau `val_loss` tidak membaik selama 5 epoch (early stopping).

### A5. Latih model usulan M2 (ResNet50-GLCM)
Berjalan otomatis setelah M1. Hasil perbandingan M1 vs M2 muncul di bagian bawah notebook dan tersimpan di `perbandingan_M1_M2.csv`.
Opsional: M3 dan M4 ada di sel terakhir. Hapus tanda `#` lalu jalankan sel itu.

### A6. Baca hasilnya
Setiap model punya folder `hasil_Mx/`. Yang terpenting adalah `ringkasan_hasil.json`:

| Kunci | Arti |
|---|---|
| `akurasi_uji` | proporsi prediksi benar pada data uji |
| `f1_macro` | rata-rata F1 dari 30 kelas (lebih adil daripada akurasi) |
| `epoch_terbaik` | epoch dengan val_loss terendah (bobot ini yang disimpan) |
| `pasangan_kelas_tertukar_terbanyak` | [kelas asli, prediksi, jumlah] |

Cara membaca `kurva_pelatihan.png`:
- Kurva train dan validasi sama-sama turun dan berdekatan: **baik**.
- Train terus turun tetapi validasi naik: **overfitting**. Bobot terbaik tetap tersimpan sebelum overfitting terjadi.

Analisis utama laporan: **bandingkan `f1_macro` dan `akurasi_uji` M2 dengan M1.**

### A7. Unduh hasil ke laptop
Dari Drive, unduh lalu taruh di folder proyek laptop:
```
hasil_M2/best.keras                       (wajib, model untuk aplikasi)
hasil_M2/ringkasan_hasil.json             (opsional, menampilkan akurasi di aplikasi)
preprocessing_output/scaler_glcm.joblib   (wajib, untuk fitur GLCM)
preprocessing_output/class_names.json     (wajib, urutan kelas)
```
Strukturnya harus persis `hasil_M2\best.keras`, bukan `best.keras` langsung di folder proyek.

---

## Bagian B: Implementasi aplikasi (laptop)

### B1. Pahami cara kerjanya
```
Browser (app/static/index.html)
   │  1. pengguna menulis/unggah aksara → klik "Kenali"
   │  2. citra dikirim:  POST /api/predict
   ▼
Backend FastAPI (app/main.py)
   │  3. validasi berkas (≤5 MB, harus citra)
   │  4. preprocess_gray()   → crop, padding, 224×224   (fungsi yang SAMA dengan pelatihan)
   │  5. cabang CNN: normalize_for()   |   cabang GLCM: glcm_features() + scaler
   │  6. model.predict() → probabilitas 30 kelas
   │  7. ambil 5 teratas → kirim JSON
   ▼
Browser menampilkan aksara, nama baca, keyakinan, top-5
```
Kunci integrasinya: **aplikasi memakai fungsi preprocessing yang sama dengan pelatihan**. Kalau berbeda (misalnya lupa crop atau normalisasi), akurasi di aplikasi akan jauh lebih rendah daripada saat pelatihan.

Bagian penting di `app/main.py`:
- `lifespan()`: memuat model, scaler, dan daftar kelas satu kali saat server mulai.
- `_predict()`: menyiapkan input dua cabang lalu memanggil model.
- `predict()`: endpoint `/api/predict` (validasi → preprocessing → prediksi → JSON).
- `AKSARA`: memetakan label (`ka`) ke huruf Sunda Unicode (ᮊ) untuk ditampilkan.

### B2. Instal library
Di PowerShell, dari folder proyek:
```powershell
conda activate pbo
cd "C:\Users\Anggia Putri Andini\Downloads\Preprocessing_ResNet50_GLCM"
pip install -r requirements.txt
```
TensorFlow butuh Python 3.10–3.12. Cek dengan `python --version`. Kalau versinya 3.13 dan instalasi TensorFlow gagal, buat env baru:
```powershell
conda create -n aksara python=3.11 -y
conda activate aksara
pip install -r requirements.txt
```

### B3. Jalankan server
```powershell
uvicorn app.main:app --port 8000
```
Tunggu sampai muncul:
```
Model M2 (resnet50, GLCM=True) siap.
Uvicorn running on http://127.0.0.1:8000
```
Untuk memakai model lain (misalnya baseline M1):
```powershell
$env:MODEL="M1"; uvicorn app.main:app --port 8000
```
Hentikan server dengan **Ctrl + C**.

### B4. Coba aplikasinya
1. Buka browser ke **http://localhost:8000**.
2. Tab **Gambar langsung**: tulis satu aksara, misalnya ᮊ (ka), lalu klik **Kenali**.
3. Tab **Unggah foto**: pilih satu citra dari `dataset_gabung_fix\` atau foto tulisan sendiri.
4. Periksa panel hasil: aksara, keyakinan, top-5, dan thumbnail "citra setelah preprocessing".
5. Cek API langsung: buka http://localhost:8000/docs. Ini halaman dokumentasi otomatis FastAPI, tempat `/api/predict` bisa dicoba tanpa frontend.

### B5. Ambil bahan laporan
- Tangkapan layar aplikasi dengan hasil prediksi asli. Gambar ini menggantikan `gambar_laporan/11_mockup_hasil_ILUSTRASI.png`, yang angkanya hanya ilustrasi.
- Untuk Bab 5 (Pengujian): coba beberapa citra dari data uji (`preprocessing_output/split_test.csv`) dan tulisan tangan baru, lalu catat benar/salah dan keyakinannya.

---

## Masalah yang sering muncul

| Pesan / gejala | Penyebab | Solusi |
|---|---|---|
| `hasil_M2\best.keras tidak ditemukan` | model belum dilatih atau belum diunduh | ulangi A5–A7, cek nama folder |
| `No such file ... class_names.json` / `scaler_glcm.joblib` | folder `preprocessing_output` belum diunduh | unduh dari Drive (A7) |
| `ModuleNotFoundError: No module named 'config'` | uvicorn dijalankan dari folder yang salah | `cd` ke folder proyek dulu |
| `Error loading ASGI app. Could not import module "app.main"` | folder `app\` tidak ada atau salah tempat | pastikan ada `app\main.py` sejajar `config.py` |
| `address already in use` | server lama masih jalan | tutup terminal lama, atau pakai `--port 8001` |
| Halaman terbuka tapi "Backend tidak dapat dihubungi" | server mati atau error | lihat pesan error di terminal uvicorn |
| Error versi saat memuat `best.keras` | versi TensorFlow laptop berbeda dari Colab | cek `tf.__version__` di Colab, lalu `pip install tensorflow==<versi itu>` |
| GPU tidak terdeteksi di Colab | runtime belum diganti | Runtime → Change runtime type → T4 GPU |
| Colab terputus di tengah pelatihan | batas waktu sesi | jalankan ulang hanya model yang belum selesai; preprocessing tidak perlu diulang |
