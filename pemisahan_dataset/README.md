# Pemisahan Foto Formulir Tulisan Tangan Aksara Sunda

Folder ini berisi kode untuk memisahkan citra lembar formulir manual aksara Sunda menjadi file gambar per karakter.

## Struktur File
* `split_dataset_manual.py`: Script Python mandiri. Jalankan langsung via terminal.
* `pemisahan_dataset.ipynb`: Jupyter Notebook interaktif (lengkap dengan visualisasi dan preview potongan citra).

## Cara Menjalankan

### Opsi 1: Menjalankan Script Python (`.py`)
```bash
python pemisahan_dataset/split_dataset_manual.py
```

### Opsi 2: Menjalankan Notebook (`.ipynb`)
Buka `pemisahan_dataset/pemisahan_dataset.ipynb` di VS Code / Jupyter Lab, lalu jalankan semua sel secara berurutan.

Opsi script: `--input`, `--output`, dan `--keep` (jangan hapus output lama). Tanpa `--keep`, folder output dihapus dulu agar penomoran tidak dobel.

## Format Input
* `foto_formulir/<kelas>/<foto>.jpg|jpeg|png`, nama subfolder = nama kelas (`a`, `i`, `sa`, `xa`, ...).
* Satu foto = satu lembar formulir tabel 8 kolom × 10 baris. Nama file bebas.
* Foto boleh portrait atau landscape, boleh sedikit miring/melengkung.

## Spesifikasi Pemrosesan
1. Orientasi otomatis: foto landscape diputar, lalu dicek header (AKSARA/HALAMAN) harus di atas tabel.
2. Tabel dideteksi lalu diluruskan (perspective transform).
3. Batas tiap kotak dicari lokal, sehingga kertas melengkung tetap terpotong pas.
4. Sisa garis tabel dibuang; hanya goresan aksara yang ditempel di kanvas putih `224x224` dengan padding.
5. Hasil disimpan ke `dataset_tulisan_tangan/<kelas>/<kelas>_<nnn>.jpg`. Kotak kosong dilaporkan di output terminal.
