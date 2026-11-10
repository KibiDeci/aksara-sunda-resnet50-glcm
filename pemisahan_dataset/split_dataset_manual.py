"""
Pemotong foto formulir tulisan tangan aksara Sunda.

Setiap foto lembar berisi tabel 8 kolom x 10 baris (80 aksara). Program:
1. Menegakkan orientasi lembar (portrait, header di atas).
2. Mendeteksi tabel dan meluruskan perspektifnya.
3. Mencari garis tiap kotak secara lokal (tahan terhadap kertas melengkung).
4. Memotong aksara, membuang sisa garis, menaruh di kanvas putih 224x224.

Struktur input : foto_formulir/<kelas>/<foto>.jpg|jpeg|png
Struktur output: dataset_tulisan_tangan/<kelas>/<kelas>_<nnn>.jpg
"""
from pathlib import Path
import argparse
import shutil

import cv2
import numpy as np

ROWS, COLS = 10, 8
CELL = 120                      # ukuran kotak setelah tabel diluruskan (px)
SEARCH = 30                     # jendela pencarian garis lokal (px)
EXTS = {".jpg", ".jpeg", ".png"}


def line_masks(gray, min_len=None):
    """Mask garis horizontal dan vertikal tabel.

    min_len harus lebih panjang dari goresan aksara terpanjang, agar goresan
    lurus pada aksara tidak dianggap garis tabel.
    """
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 25, 10)
    h, w = gray.shape
    len_h = min_len or max(15, w // 25)
    len_v = min_len or max(15, h // 25)
    # Penebalan tipis tegak lurus garis agar garis yang sedikit miring tetap utuh
    bw_h = cv2.dilate(bw, np.ones((3, 1), np.uint8))
    bw_v = cv2.dilate(bw, np.ones((1, 3), np.uint8))
    lh = cv2.morphologyEx(bw_h, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (len_h, 1)))
    lv = cv2.morphologyEx(bw_v, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, len_v)))
    return bw, lh & bw, lv & bw


def largest_contours(gray, n=2):
    _, lh, lv = line_masks(gray)
    grid = cv2.dilate(cv2.add(lh, lv), np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(grid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(cnts, key=cv2.contourArea, reverse=True)[:n]


def orient_upright(img):
    """Putar lembar menjadi portrait dengan header (AKSARA/HALAMAN) di atas tabel."""
    h, w = img.shape[:2]
    if w > h:
        img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cnts = largest_contours(gray, 2)
    if len(cnts) == 2:
        table_y = cv2.boundingRect(cnts[0])[1]
        header_y = cv2.boundingRect(cnts[1])[1]
        if header_y > table_y:
            img = cv2.rotate(img, cv2.ROTATE_180)
    return img


def table_corners(cnt):
    """Empat sudut tabel: kiri-atas, kanan-atas, kanan-bawah, kiri-bawah."""
    pts = cnt.reshape(-1, 2).astype(np.float32)
    s, d = pts.sum(1), pts[:, 0] - pts[:, 1]
    return np.array([pts[s.argmin()], pts[d.argmax()], pts[s.argmax()], pts[d.argmin()]],
                    dtype=np.float32)


def warp_table(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cnts = largest_contours(gray, 1)
    if not cnts:
        raise ValueError("tabel tidak terdeteksi")
    src = table_corners(cnts[0])
    W, H = COLS * CELL, ROWS * CELL
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (W, H), borderValue=(255, 255, 255))


def find_line(profile, expected, lo, hi):
    """Posisi garis = puncak profil di sekitar posisi perkiraan."""
    a, b = max(lo, expected - SEARCH), min(hi, expected + SEARCH + 1)
    seg = profile[a:b]
    if seg.size == 0 or seg.max() == 0:
        return expected
    return a + int(seg.argmax())


def cell_boxes(warped):
    """Batas tiap kotak dicari lokal per baris/kolom, sehingga lengkungan kertas tertangani."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # Garis tabel melintasi beberapa kotak, goresan aksara tidak
    _, lh, lv = line_masks(gray, min_len=int(CELL * 1.5))
    H, W = gray.shape
    m = CELL // 6
    boxes = {}
    for r in range(ROWS):
        for c in range(COLS):
            x0, x1 = c * CELL, (c + 1) * CELL
            y0, y1 = r * CELL, (r + 1) * CELL
            prof_h = lh[:, x0 + m:x1 - m].sum(1)
            prof_v = lv[y0 + m:y1 - m, :].sum(0)
            top = find_line(prof_h, y0, 0, H)
            bot = find_line(prof_h, y1, 0, H)
            left = find_line(prof_v, x0, 0, W)
            right = find_line(prof_v, x1, 0, W)
            boxes[r, c] = (top, bot, left, right)
    return boxes, lh, lv


def isolate_character(cell_bgr, line_mask, pad_ratio=0.15, out_size=(224, 224)):
    """Ambil goresan aksara dalam kotak, buang sisa garis tabel, pusatkan di kanvas putih."""
    gray = cv2.cvtColor(cell_bgr, cv2.COLOR_BGR2GRAY)
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    ink[cv2.dilate(line_mask, np.ones((5, 5), np.uint8)) > 0] = 0

    h, w = ink.shape
    n, _, st, _ = cv2.connectedComponentsWithStats(ink)
    boxes = []
    for i in range(1, n):
        x, y, bw, bh, area = st[i]
        if area < 15:
            continue
        touches = x <= 2 or y <= 2 or x + bw >= w - 2 or y + bh >= h - 2
        # Garis tabel: memanjang dan tipis (toleransi tebal untuk garis yang sedikit miring)
        thin = min(bw, bh) <= 0.15 * min(w, h) and max(bw, bh) >= 4 * min(bw, bh)
        if touches and (thin or area < 80):
            continue        # sisa garis tabel di tepi kotak
        if thin and max(bw, bh) >= 0.6 * max(w, h):
            continue        # garis tabel lepas yang melintang kotak
        boxes.append((x, y, x + bw, y + bh))

    canvas_bg = np.full((*out_size[::-1], 3), 255, np.uint8)
    if not boxes:
        return canvas_bg, False

    x0 = min(b[0] for b in boxes); y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes); y1 = max(b[3] for b in boxes)
    # Tempel goresan saja di atas latar putih, agar sisa garis/bayangan tidak ikut
    crop = cell_bgr[y0:y1, x0:x1].copy()
    crop[ink[y0:y1, x0:x1] == 0] = 255
    ch, cw = crop.shape[:2]
    side = int(max(ch, cw) / (1 - 2 * pad_ratio))
    sq = np.full((side, side, 3), 255, np.uint8)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    sq[oy:oy + ch, ox:ox + cw] = crop
    return cv2.resize(sq, out_size, interpolation=cv2.INTER_AREA), True


def split_sheet(img_bgr, inset=4):
    """Kembalikan list 80 citra aksara (urut baris lalu kolom) dan jumlah kotak kosong."""
    warped = warp_table(orient_upright(img_bgr))
    boxes, lh, lv = cell_boxes(warped)
    lines = cv2.add(lh, lv)
    out, empty = [], 0
    for r in range(ROWS):
        for c in range(COLS):
            top, bot, left, right = boxes[r, c]
            ys, ye, xs, xe = top + inset, bot - inset, left + inset, right - inset
            char, ok = isolate_character(warped[ys:ye, xs:xe], lines[ys:ye, xs:xe])
            empty += not ok
            out.append(char)
    return out, empty


def process_dataset(input_dir, output_dir, clean=True):
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    if clean and output_dir.exists():
        shutil.rmtree(output_dir)
    files = sorted(p for p in input_dir.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        print(f"Tidak ada citra di '{input_dir}'.")
        return
    print(f"Ditemukan {len(files)} lembar.")

    counter = {}
    for i, f in enumerate(files, 1):
        cls = f.parent.name
        img = cv2.imread(str(f))
        if img is None:
            print(f"[{i}/{len(files)}] gagal dibaca: {f}")
            continue
        try:
            chars, empty = split_sheet(img)
        except ValueError as e:
            print(f"[{i}/{len(files)}] {f}: {e}")
            continue
        target = output_dir / cls
        target.mkdir(parents=True, exist_ok=True)
        for ch in chars:
            counter[cls] = counter.get(cls, 0) + 1
            cv2.imwrite(str(target / f"{cls}_{counter[cls]:03d}.jpg"), ch)
        note = f" ({empty} kotak kosong)" if empty else ""
        print(f"[{i}/{len(files)}] {cls}/{f.name} -> {len(chars)} aksara{note}")

    print("\nTotal per kelas:")
    for cls, n in sorted(counter.items()):
        print(f"  {cls}: {n}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description="Pisahkan foto formulir menjadi citra per aksara.")
    ap.add_argument("--input", default=root / "foto_formulir")
    ap.add_argument("--output", default=root / "dataset_tulisan_tangan")
    ap.add_argument("--keep", action="store_true", help="jangan hapus folder output lama")
    a = ap.parse_args()
    process_dataset(a.input, a.output, clean=not a.keep)
