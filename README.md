# 🔐📸 License Plate Detect & OCR

**Detekcja tablic rejestracyjnych (YOLOv11) + OCR (DeepSeek-OCR)** w jednym przebiegu: wykrywa tablice, wycina je i rozpoznaje tekst.

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Ultralytics](https://img.shields.io/badge/Ultralytics-YOLOv11-111?logo=ultralytics)
![Transformers](https://img.shields.io/badge/HF-Transformers-FFAE33?logo=huggingface)
![OS](https://img.shields.io/badge/OS-Windows-blue?logo=windows)
![GPU](https://img.shields.io/badge/GPU-Optional-2ea44f)

---

## ✨ Co robi?
- 🚗 **Wykrywa** tablice na obrazach/wideo (YOLO, własne wagi `.pt`)
- ✂️ **Wycina** tablice (`save_crop`)
- 🔤 **Rozpoznaje** tekst (DeepSeek-OCR przez `transformers`)
- 🧾 **Zapisuje** wyniki: podglądy YOLO, pliki `.txt` oraz zbiorcze **`plates_ocr.csv`**

---

## 📦 Wymagania
- Python **3.10+**
- (Opcjonalnie) CUDA dla szybszego działania
- Modele:
  - `license_plate_detector.pt` (YOLO) ➜ umieść w katalogu projektu
  - OCR: `deepseek-ai/DeepSeek-OCR` (pobierany automatycznie z HF Hub)

---

## ⚙️ Konfiguracja (w pliku)
W `lp_detect_and_ocr.py` edytuj ścieżki pod siebie:
```python
BASE_DIR = Path(r"C:\Users\ms\Desktop\suml")
WEIGHTS  = BASE_DIR / "license_plate_detector.pt"
SOURCE   = BASE_DIR / "images_cars"          # folder/plik z obrazami lub wideo

YOLO_OUT = BASE_DIR / "outputs" / "yolo"
OCR_OUT  = BASE_DIR / "outputs" / "ocr"
OCR_MODEL_NAME = "deepseek-ai/DeepSeek-OCR"

CONF = 0.25
IOU  = 0.50
IMGSZ = 640
