"""Backend aplikasi pengenalan Aksara Sunda (FastAPI).

Jalankan dari folder proyek (sejajar dengan config.py):
    uvicorn app.main:app --port 8000            # memakai model M2 (default)
    set MODEL=M1 && uvicorn app.main:app        # Windows CMD: pilih model lain
    $env:MODEL="M1"; uvicorn app.main:app       # Windows PowerShell
Lalu buka http://localhost:8000

Backend memakai modul preprocessing, GLCM, dan normalisasi yang SAMA dengan
pelatihan, sehingga citra dari pengguna diproses persis seperti data latih.
"""
import base64
import io
import json
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # folder proyek
sys.path.insert(0, str(ROOT))

import cv2
import joblib
import keras
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import UnidentifiedImageError

import config as C
from glcm_features import glcm_features
from preprocessing import gray_to_rgb, normalize_for, preprocess_gray

MODEL_NAME = os.environ.get("MODEL", "M2")
MODEL_DIR = ROOT / f"hasil_{MODEL_NAME}"
MODELS = {"M1": ("resnet50", False), "M2": ("resnet50", True),
          "M3": ("mobilenetv2", False), "M4": ("mobilenetv2", True)}
MAX_BYTES = 5 * 1024 * 1024
TOP_K = 5

# Label kelas -> karakter Unicode Aksara Sunda (blok U+1B80-U+1BBF) dan nama baca
AKSARA = {"a": "ᮃ", "i": "ᮄ", "u": "ᮅ", "ae": "ᮆ", "o": "ᮇ",
          "e": "ᮈ", "eu": "ᮉ", "ka": "ᮊ", "qa": "ᮋ", "ga": "ᮌ",
          "nga": "ᮍ", "ca": "ᮎ", "ja": "ᮏ", "za": "ᮐ", "nya": "ᮑ",
          "ta": "ᮒ", "da": "ᮓ", "na": "ᮔ", "pa": "ᮕ", "fa": "ᮖ",
          "va": "ᮗ", "ba": "ᮘ", "ma": "ᮙ", "ya": "ᮚ", "ra": "ᮛ",
          "la": "ᮜ", "wa": "ᮝ", "sa": "ᮞ", "xa": "ᮟ", "ha": "ᮠ"}
BACA = {"ae": "é", "e": "e (pepet)"}
SWARA = {"a", "i", "u", "ae", "o", "e", "eu"}

S = {}   # state yang dimuat sekali saat server mulai


@asynccontextmanager
async def lifespan(app):
    if MODEL_NAME not in MODELS:
        raise RuntimeError(f"MODEL harus salah satu dari {list(MODELS)}")
    ckpt = MODEL_DIR / "best.keras"
    if not ckpt.exists():
        raise RuntimeError(f"{ckpt} tidak ditemukan. Latih dulu: python train_model.py --model {MODEL_NAME}")
    S["arch"], S["glcm"] = MODELS[MODEL_NAME]
    S["model"] = keras.models.load_model(ckpt)
    S["classes"] = json.load(open(C.OUT_DIR / "class_names.json"))
    S["scaler"] = joblib.load(C.OUT_DIR / "scaler_glcm.joblib") if S["glcm"] else None
    rh = MODEL_DIR / "ringkasan_hasil.json"
    S["ringkasan"] = json.load(open(rh)) if rh.exists() else {}
    _predict(np.full((C.IMG_SIZE, C.IMG_SIZE), 255, np.uint8))   # pemanasan
    print(f"Model {MODEL_NAME} ({S['arch']}, GLCM={S['glcm']}) siap.")
    yield
    S.clear()


app = FastAPI(title="Pengenalan Aksara Sunda", lifespan=lifespan)


def _predict(gray: np.ndarray) -> np.ndarray:
    x = normalize_for(S["arch"], gray_to_rgb(gray)[None])
    if S["glcm"]:
        f = S["scaler"].transform(glcm_features(gray)[None]).astype(np.float32)
        x = [x, f]
    return S["model"].predict(x, verbose=0)[0]


@app.get("/api/info")
def info():
    r = S["ringkasan"]
    return {"model": MODEL_NAME, "nama": r.get("nama", MODEL_NAME), "arsitektur": S["arch"],
            "glcm": S["glcm"], "jumlah_kelas": len(S["classes"]),
            "akurasi_uji": r.get("akurasi_uji"), "f1_macro": r.get("f1_macro")}


@app.get("/api/classes")
def classes():
    return [{"label": c, "aksara": AKSARA.get(c, ""), "baca": BACA.get(c, c),
             "jenis": "swara" if c in SWARA else "ngalagena"} for c in S["classes"]]


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not data:
        raise HTTPException(400, "Berkas kosong.")
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Ukuran berkas maksimal 5 MB.")
    t0 = time.perf_counter()
    try:
        gray = preprocess_gray(io.BytesIO(data))
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(415, "Berkas bukan citra yang valid (gunakan JPG/PNG).")
    prob = _predict(gray)
    ms = (time.perf_counter() - t0) * 1000

    top = np.argsort(prob)[::-1][:TOP_K]
    ok, png = cv2.imencode(".png", gray)
    return {
        "prediksi": [{"label": S["classes"][i], "aksara": AKSARA.get(S["classes"][i], ""),
                      "baca": BACA.get(S["classes"][i], S["classes"][i]),
                      "probabilitas": round(float(prob[i]), 4)} for i in top],
        "citra_diproses": "data:image/png;base64," + base64.b64encode(png.tobytes()).decode(),
        "waktu_ms": round(ms, 1),
        "model": MODEL_NAME,
    }


app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True), name="static")


if __name__ == "__main__":
    # Memungkinkan dijalankan langsung: python app/main.py  (atau tombol Run di VS Code)
    import uvicorn
    print("Buka http://localhost:8000 di browser (Ctrl+C untuk berhenti)")
    uvicorn.run(app, host="127.0.0.1", port=8000)
