"""
Silnik do wykrywania tablic rejestracyjnych (YOLO)
oraz OCR przy użyciu modelu Florence-2.
"""

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from ultralytics import YOLO
from transformers import AutoProcessor, AutoModelForCausalLM

# --------------------------------------------------
# Konfiguracja ścieżek i identyfikatorów modeli
# --------------------------------------------------

# Katalog, w którym znajduje się ten plik
BASE_DIR = Path(__file__).resolve().parent

# Wagi wytrenowanego modelu YOLO do detekcji tablic
WEIGHTS = BASE_DIR / "license_plate_detector.pt"

# Id modelu Florence-2 (możliwy override przez zmienną środowiskową)
FLORENCE_MODEL_ID = os.environ.get(
    "FLORENCE_MODEL_ID",
    "microsoft/Florence-2-large"
)

# --------------------------------------------------
# Parametry detekcji i generacji
# --------------------------------------------------

# Rozmiar obrazu wejściowego dla YOLO
IMGSZ = 1280

# Minimalna pewność detekcji
CONF = 0.25

# Próg IoU do NMS
IOU = 0.45

# Parametry generacji tekstu (OCR)
MAX_NEW_TOKENS = 256
NUM_BEAMS = 3

# --------------------------------------------------
# Konfiguracja urządzeń (CPU / GPU)
# --------------------------------------------------

# YOLO: indeks GPU lub CPU
DEVICE_YOLO = 0 if torch.cuda.is_available() else "cpu"

# Florence-2: stringowy identyfikator urządzenia
DEVICE_LLM = "cuda" if torch.cuda.is_available() else "cpu"

# Typ danych tensora (float16 na GPU, float32 na CPU)
DTYPE_LLM = torch.float16 if torch.cuda.is_available() else torch.float32


class PlateReader:
    """
    Klasa odpowiedzialna za:
    - wykrywanie tablic rejestracyjnych (YOLO)
    - rozpoznawanie tekstu z obrazu tablicy (OCR)
    """

    def __init__(self) -> None:
        """Ładuje modele YOLO i Florence-2 do pamięci."""
        print(f"🚦 Ładowanie YOLO na: {DEVICE_YOLO}")
        self.yolo_model = YOLO(str(WEIGHTS))

        print(f"🧠 Ładowanie Florence-2: {FLORENCE_MODEL_ID}")
        self.florence_model = AutoModelForCausalLM.from_pretrained(
            FLORENCE_MODEL_ID,
            trust_remote_code=True,
            attn_implementation="eager",
            torch_dtype=DTYPE_LLM,
        ).to(DEVICE_LLM)

        # Processor odpowiada za:
        # - przygotowanie wejścia (obrazy + prompt)
        # - dekodowanie wygenerowanego tekstu
        self.florence_processor = AutoProcessor.from_pretrained(
            FLORENCE_MODEL_ID,
            trust_remote_code=True
        )

    def detect_crops(
        self,
        pil_img: Image.Image
    ) -> List[Tuple[Image.Image, float, List[int]]]:
        """
        Wykrywa tablice rejestracyjne na obrazie
        i zwraca ich wycinki (cropy).

        :param pil_img: Obraz wejściowy PIL
        :return: Lista krotek:
                 (obraz tablicy, pewność detekcji, bounding box)
        """
        # Konwersja PIL -> numpy (wymagane przez YOLO)
        np_img = np.array(pil_img)

        # Uruchomienie predykcji YOLO
        results = self.yolo_model.predict(
            source=np_img,
            imgsz=IMGSZ,
            conf=CONF,
            iou=IOU,
            device=DEVICE_YOLO,
            verbose=False
        )

        crops: List[Tuple[Image.Image, float, List[int]]] = []

        # Brak wykryć
        if not results or not results[0].boxes:
            return crops

        result = results[0]
        height, width = np_img.shape[:2]

        # Iteracja po wykrytych bounding boxach
        for box in result.boxes:
            # Współrzędne [x1, y1, x2, y2]
            coords = box.xyxy[0].tolist()

            # Pewność predykcji
            confidence = float(box.conf.item())

            # Przycięcie współrzędnych do granic obrazu
            x1, y1, x2, y2 = [
                int(
                    max(
                        0,
                        min(value, width if index % 2 == 0 else height)
                    )
                )
                for index, value in enumerate(coords)
            ]

            # Wycięcie fragmentu obrazu z tablicą
            crop_img = pil_img.crop((x1, y1, x2, y2))

            crops.append((crop_img, confidence, [x1, y1, x2, y2]))

        # Sortowanie od najbardziej pewnej detekcji
        crops.sort(key=lambda item: item[1], reverse=True)
        return crops

    def run_ocr(self, pil_img: Image.Image) -> str:
        """
        Wykonuje OCR na obrazie tablicy
        przy użyciu modelu Florence-2.

        :param pil_img: Obraz tablicy
        :return: Rozpoznany tekst
        """
        # Prompt sterujący trybem OCR
        prompt = "<OCR>"

        # Przygotowanie wejścia dla modelu
        inputs = self.florence_processor(
            text=prompt,
            images=pil_img,
            return_tensors="pt"
        )

        # Przeniesienie tensorów na właściwe urządzenie
        inputs = {
            key: value.to(DEVICE_LLM)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in inputs.items()
        }

        # Generacja tekstu bez gradientów
        with torch.inference_mode():
            generated_ids = self.florence_model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=MAX_NEW_TOKENS,
                num_beams=NUM_BEAMS,
                do_sample=False,
            )

        # Dekodowanie wygenerowanego tekstu
        generated_text = self.florence_processor.batch_decode(
            generated_ids,
            skip_special_tokens=False
        )[0]

        # Post-processing wyniku OCR
        parsed = self.florence_processor.post_process_generation(
            generated_text,
            task=prompt,
            image_size=(pil_img.width, pil_img.height)
        )

        # Wyciągnięcie tekstu z różnych możliwych formatów
        if isinstance(parsed, dict):
            for value in parsed.values():
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if (
                    isinstance(value, list)
                    and value
                    and isinstance(value[0], str)
                ):
                    return value[0].strip()

        return str(parsed)
