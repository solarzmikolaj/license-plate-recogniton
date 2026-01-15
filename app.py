"""
Aplikacja webowa Gradio do odczytu tablic rejestracyjnych.
Łączy detekcję YOLO z OCR opartym o model Florence-2.
"""

from typing import List, Tuple, Optional

import gradio as gr
from PIL import Image

from model.engine import PlateReader

# --------------------------------------------------
# Inicjalizacja silnika ML
# --------------------------------------------------
# Obiekt tworzony raz przy starcie aplikacji.
# Modele YOLO i Florence-2 są trzymane w pamięci,
# aby uniknąć kosztownego przeładowywania.
READER = PlateReader()


def process_image(
    pil_img: Optional[Image.Image]
) -> Tuple[str, Optional[List[Image.Image]], List[List[str]]]:
    """
    Przetwarza obraz wejściowy:
    - wykrywa tablice rejestracyjne,
    - wykonuje OCR dla każdej z nich,
    - wybiera najlepszy wynik.

    :param pil_img: Obraz wejściowy w formacie PIL
    :return:
        - główny rozpoznany tekst,
        - lista obrazów tablic (galeria),
        - tabela wyników (idx, pewność, tekst)
    """
    # Brak obrazu wejściowego
    if pil_img is None:
        return "Brak obrazu.", None, []

    # Detekcja tablic rejestracyjnych
    crops = READER.detect_crops(pil_img)
    if not crops:
        return "Nie wykryto tablic rejestracyjnych.", None, []

    # Dane wyjściowe dla UI
    gallery_items: List[Image.Image] = []
    rows: List[List[str]] = []

    # Najlepszy wynik (najwyższa pewność YOLO + poprawny OCR)
    best_text: Optional[str] = None
    best_confidence = -1.0

    # Iteracja po wykrytych tablicach
    for index, (crop_img, confidence, _) in enumerate(crops, start=1):
        try:
            # OCR pojedynczej tablicy
            text = READER.run_ocr(crop_img)
        except RuntimeError as exc:
            # Obsługa błędu OCR (np. brak pamięci GPU)
            text = f"Błąd OCR: {exc}"

        # Dane do galerii i tabeli
        gallery_items.append(crop_img)
        rows.append([str(index), f"{confidence:.3f}", text])

        # Aktualizacja najlepszego wyniku
        if (
            confidence > best_confidence
            and text
            and not text.startswith("Błąd")
        ):
            best_confidence = confidence
            best_text = text

    # Tekst główny wyświetlany użytkownikowi
    headline = best_text or rows[0][2]
    return headline, gallery_items, rows


def build_ui() -> gr.Blocks:
    """
    Buduje interfejs użytkownika Gradio.

    :return: Obiekt gr.Blocks reprezentujący UI
    """
    with gr.Blocks(title="LPR - YOLO + Florence-2") as demo:
        # Nagłówek aplikacji
        gr.Markdown("# 📸🔎 License Plate Reader")

        # Sekcja wgrywania obrazu
        with gr.Row():
            image_input = gr.Image(
                type="pil",
                label="Wgraj zdjęcie"
            )

        # Przycisk uruchamiający analizę
        with gr.Row():
            run_button = gr.Button(
                "Uruchom analizę",
                variant="primary"
            )

        # Pole tekstowe z najlepszym rozpoznanym numerem
        output_text = gr.Textbox(
            label="Zidentyfikowany numer",
            interactive=False
        )

        # Galeria wykrytych tablic
        gallery = gr.Gallery(
            label="Wykryte tablice",
            columns=4,
            height=250
        )

        # Tabela z wynikami OCR
        table = gr.Dataframe(
            headers=["#", "Pewność (YOLO)", "Tekst (OCR)"],
            interactive=False
        )

        # Powiązanie przycisku z funkcją przetwarzania
        run_button.click(
            fn=process_image,
            inputs=image_input,
            outputs=[output_text, gallery, table]
        )

    return demo


# --------------------------------------------------
# Punkt wejścia aplikacji
# --------------------------------------------------
if __name__ == "__main__":
    APP = build_ui()
    APP.launch()
