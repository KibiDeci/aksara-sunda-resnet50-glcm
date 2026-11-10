# Tutorial Implementasi: Model ResNet50-GLCM dan Aplikasi di VS Code

Dua bagian:
- **Bagian A**: melatih model (M1 baseline, M2 model usulan) di VS Code.
- **Bagian B**: menjalankan aplikasi web lokal memakai model hasil pelatihan.

Prasyarat: repositori sudah berisi folder `dataset\`, `eda_output\`, dan semua file skrip.

---

## Bagian A: Implementasi model di VS Code

Pelatihan dapat dilakukan di VS Code lewat Terminal atau Jupyter Notebook interaktif.

### A1. Buka proyek di VS Code
1. Buka aplikasi VS Code.
2. Pilih menu **File > Open Folder...**, lalu arahkan ke root folder repositori ini.
3. Buka integrated terminal di VS Code (**Ctrl + `** atau menu **Terminal > New Terminal**).

### A2. Siapkan virtual environment
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```
Pastikan Python versi 3.10–3.12.

### A3. Preprocessing
Jalankan di Terminal VS Code:
```bash
python run_preprocessing.py
```
Ringkasan akan menampilkan:
```
Split akhir : 1,244 / 266 / 266 = 1,776 citra, 30 kelas
```
Citra dipotong ke area karakter, diberi padding persegi, di-resize 224×224, duplikat dibuang, dan 32 fitur GLCM dihitung ke `preprocessing_output/`.

### A4. Latih baseline M1 (ResNet50)
```bash
python train_model.py --model M1
```
- **Tahap 1:** ResNet50 dibekukan, hanya lapisan klasifikasi yang belajar.
- **Tahap 2:** 30 lapisan terakhir ikut dilatih (fine-tuning) dengan learning rate kecil.
- Pelatihan berhenti otomatis kalau `val_loss` tidak membaik selama 5 epoch (early stopping).

### A5. Latih model usulan M2 (ResNet50-GLCM)
```bash
python train_model.py --model M2
```
Model menggabungkan fitur CNN ResNet50 dengan 32 fitur GLCM.
*(Opsional: untuk melatih M3 dan M4, jalankan dengan `--model M3` dan `--model M4`).*

### A6. Baca hasilnya
Setiap model punya folder `hasil_Mx/`. Yang terpenting adalah `ringkasan_hasil.json`:

| Kunci | Arti |
|---|---|
| `akurasi_uji` | proporsi prediksi benar pada data uji |
| `f1_macro` | rata-rata F1 dari 30 kelas (lebih adil daripada akurasi) |
| `epoch_terbaik` | epoch dengan val_loss terendah (bobot ini yang disimpan) |
| `pasangan_kelas_tertukar_terbanyak` | [kelas asli, prediksi, jumlah] |

Bandingkan `f1_macro` dan `akurasi_uji` M2 dengan M1. Semua grafik loss dan confusion matrix tersimpan di masing-masing folder `hasil_M*/`.

---

## Bagian B: Implementasi aplikasi (VS Code)

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

### B2. Jalankan server di Terminal VS Code
```powershell
python app/main.py
```
Tunggu sampai muncul:
```
Model M2 (resnet50, GLCM=True) siap.
Uvicorn running on http://127.0.0.1:8000
```
Untuk memakai model lain (misalnya baseline M1):
```powershell
$env:MODEL="M1"; python app/main.py
```
Hentikan server dengan **Ctrl + C**.

### B3. Coba aplikasinya
1. Buka browser ke **http://localhost:8000**.
2. Tab **Gambar langsung**: tulis satu aksara, misalnya ᮊ (ka), lalu klik **Kenali**.
3. Tab **Unggah foto**: pilih satu citra dari folder `dataset/` atau foto tulisan sendiri.
4. Periksa panel hasil: aksara, keyakinan, top-5, dan thumbnail "citra setelah preprocessing".

---

## Masalah yang sering muncul

| Pesan / gejala | Penyebab | Solusi |
|---|---|---|
| `hasil_M2\best.keras tidak ditemukan` | model belum dilatih | Jalankan `python train_model.py --model M2` |
| `No such file ... class_names.json` / `scaler_glcm.joblib` | file preprocessing belum ada | Jalankan `python run_preprocessing.py` |
| `ModuleNotFoundError: No module named 'config'` | dijalankan dari folder yang salah | Pastikan working directory berada di root proyek |
| `address already in use` | port 8000 masih dipakai | Hentikan proses lama atau ubah port |
| TensorFlow gagal diinstal | versi Python tidak cocok | Gunakan Python 3.10–3.12 |
