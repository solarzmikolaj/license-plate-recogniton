# lp_detect_and_ocr.py
from ultralytics import YOLO
from transformers import AutoModel, AutoTokenizer
import torch
from pathlib import Path
import csv
import json
import os
import re

# =========================
# KONFIGURACJA
# =========================
BASE_DIR = Path(r"C:\Users\ms\Desktop\suml")     # katalog projektu
WEIGHTS = BASE_DIR / "license_plate_detector.pt" # YOLOv11 *.pt
SOURCE = BASE_DIR / "images_cars"                # plik lub folder z obrazami/wideo

YOLO_OUT = BASE_DIR / "outputs" / "yolo"         # YOLO: bbox + crops
OCR_OUT = BASE_DIR / "outputs" / "ocr"           # OCR: txt + CSV

OCR_MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

CONF = 0.25
IOU = 0.50
IMGSZ = 640

# (opcjonalnie) wycisz ostrzeżenie o symlinkach na Windows
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")


# =========================
# HELPERY (OCR)
# =========================
def _extract_text_from_result(res):
    """Spróbuj wyciągnąć tekst bezpośrednio ze zwrotki infer()."""
    if isinstance(res, dict):
        for k in ("text", "output", "result", "image"):
            v = res.get(k)
            if isinstance(v, str) and v.strip():
                return v
        segs = res.get("segments")
        if isinstance(segs, list):
            seg_text = "\n".join(seg.get("text", "") for seg in segs)
            if seg_text.strip():
                return seg_text
    return None


def _read_text_from_saved_files(out_dir: Path, stem: str):
    """Jeśli infer() nie zwróci tekstu, poszukaj .md/.txt zapisanych przez model."""
    candidates = []
    for ext in (".md", ".txt"):
        p = out_dir / f"{stem}{ext}"
        if p.exists():
            candidates.append(p)
    if not candidates:
        for p in out_dir.glob(f"*{stem}*"):
            if p.suffix.lower() in {".md", ".txt"}:
                candidates.append(p)
    if not candidates:
        return None
    p = max(candidates, key=lambda fp: fp.stat().st_mtime)
    try:
        return p.read_text(encoding="utf-8")
    except Exception:
        return None


def _clean_plate(text: str) -> str:
    """Uprość do formatu tablicy: usuwaj spacje/nowe linie, zostaw A-Z, 0-9 i '-'."""
    if not text:
        return ""
    text = re.sub(r"\s+", "", text.upper())
    text = re.sub(r"[^A-Z0-9\-]", "", text)
    return text


# =========================
# 1️⃣ DETEKCJA YOLO
# =========================
def yolo_detect_and_crop():
    YOLO_OUT.mkdir(parents=True, exist_ok=True)
    device_yolo = 0 if torch.cuda.is_available() else "cpu"
    print(f"🚦 YOLO device: {device_yolo}")

    model = YOLO(str(WEIGHTS))
    model.predict(
        source=str(SOURCE),
        imgsz=IMGSZ,
        conf=CONF,
        iou=IOU,
        device=device_yolo,
        save=True,            # zapisze obrazy z bboxami
        save_txt=True,        # YOLO labels
        save_conf=True,       # confidence
        save_crop=True,       # WYCIĘTE tablice
        project=str(YOLO_OUT),
        name="pred",
        exist_ok=True
    )

    crops_root = YOLO_OUT / "pred" / "crops"
    if not crops_root.exists():
        print("⚠️  Brak wykrytych tablic (folder crops nie istnieje).")
        return []

    crop_paths = [p for p in crops_root.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    print(f"🧩 Znaleziono {len(crop_paths)} wycinków z tablicami.")
    return crop_paths


# =========================
# 2️⃣ OCR (DeepSeek) — z Twoim dokładnym wywołaniem infer()
# =========================
def run_ocr_on_crops(crop_paths):
    OCR_OUT.mkdir(parents=True, exist_ok=True)
    device_ocr = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🔧 OCR device: {device_ocr}")

    print("📥 Ładowanie tokenizera i modelu (DeepSeek-OCR)...")
    tokenizer = AutoTokenizer.from_pretrained(OCR_MODEL_NAME, trust_remote_code=True)

    # FlashAttention (opcjonalne; na Windows często niedostępne)
    use_flash = False
    if device_ocr == "cuda":
        try:
            import flash_attn  # noqa: F401
            use_flash = True
            print("✅ FlashAttention dostępny.")
        except Exception:
            print("⚠️ FlashAttention niedostępny, używam standardowej implementacji.")

    attn_impl = "flash_attention_2" if use_flash else "eager"

    model = AutoModel.from_pretrained(
        OCR_MODEL_NAME,
        trust_remote_code=True,
        use_safetensors=True,
        _attn_implementation=attn_impl
    ).to(device_ocr).eval()

    # Twój output_path i dokładne parametry infer()
    output_path = str(OCR_OUT)  # katalog wynikowy
    csv_path = OCR_OUT / "plates_ocr.csv"
    rows = [("crop_path", "plate")]

    print("\n🔠 Rozpoczynam OCR wycinków...\n")

    for c in crop_paths:
        try:
            # ⬇️ dokładnie Twoje wywołanie infer()
            res = model.infer(
                tokenizer,
                prompt="Get text from image",
                image_file=str(c),
                output_path=output_path,
                base_size=1024,
                image_size=640,
                crop_mode=True,
                save_results=True,
                test_compress=True
            )

            # 1) spróbuj z obiektu zwrotnego
            text = _extract_text_from_result(res)

            # 2) jeśli brak – doładuj z plików zapisanych przez model
            if not text:
                text = _read_text_from_saved_files(OCR_OUT, c.stem)

            # 3) jeśli nadal brak – zapisz RAW i przejdź dalej
            if not text:
                raw_json = OCR_OUT / f"{c.stem}.deepseek.json"
                raw_json.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
                print(f"📸 {c.name} → 🏷️ ❌ brak odczytu (RAW: {raw_json.name})")
                rows.append((str(c), ""))
                continue

            clean_text = _clean_plate(text)

            if not clean_text:
                print(f"📸 {c.name} → 🏷️ ❌ brak odczytu")
                rows.append((str(c), ""))
            else:
                print(f"📸 {c.name} → 🏷️ {clean_text}")
                rows.append((str(c), clean_text))

            # zapis .txt
            (OCR_OUT / f"{c.stem}.txt").write_text(clean_text, encoding="utf-8")

        except torch.cuda.OutOfMemoryError:
            print(f"💥 CUDA OOM – pomijam {c.name}")
            rows.append((str(c), ""))
        except Exception as e:
            print(f"❌ Błąd OCR przy {c.name}: {e}")
            rows.append((str(c), ""))

    # CSV z podsumowaniem
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)

    print(f"\n📑 Zapisano CSV: {csv_path}")


# =========================
# MAIN PIPELINE
# =========================
def main():
    crops = yolo_detect_and_crop()
    if not crops:
        print("🏁 Koniec: brak wycinków do OCR.")
        return

    run_ocr_on_crops(crops)

    print("\n🎉 Skończone!")
    print("📂 YOLO wyniki:", YOLO_OUT / "pred")
    print("📂 OCR wyniki:", OCR_OUT)


if __name__ == "__main__":
    main()
