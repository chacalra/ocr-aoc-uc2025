#!/usr/bin/env python3
import os
import csv

from OCR import (
    preprocess_image,
    ocr_with_paddleocr,
    ocr_with_tesseract,
    post_process,
    calculate_accuracy,
)

# Directorio de imágenes de prueba y ground truth
DATA_DIR = "/Users/chacalra/Universidaa/9no/SE2/Code/Imagenes"
# Nombre del CSV de salida
OUTPUT_CSV = "benchmark_postprocess.csv"

# Modos a evaluar
MODES = [
    "paddle",      # solo texto con PaddleOCR
    "tesseract"    # solo texto con Tesseract (con preprocesamiento)
]


def run_raw(mode, image_path):
    """
    Ejecuta OCR sin post-procesamiento. Devuelve el texto bruto.
    """
    if mode == "paddle":
        text = ocr_with_paddleocr(image_path)
    elif mode == "tesseract":
        pre = preprocess_image(image_path)
        text = ocr_with_tesseract(pre)
    else:
        raise ValueError(f"Modo desconocido: {mode}")
    return text.strip()


def main():
    # Prepara CSV
    with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Columnas deseadas
        writer.writerow([
            "imagen",
            "modo",
            "ground-truth",
            "texto_bruto",
            "prec_raw",
            "texto_post",
            "prec_post",
            "delta_prec"
        ])

        for fname in sorted(os.listdir(DATA_DIR)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(DATA_DIR, fname)
            gt_path = os.path.splitext(img_path)[0] + ".txt"
            if not os.path.exists(gt_path):
                print(f"⚠️  No ground-truth para {fname}, saltando")
                continue

            # Lee ground-truth
            with open(gt_path, "r", encoding="utf-8") as f:
                ground_truth = f.read().strip()

            for mode in MODES:
                # OCR sin post-proceso
                raw_text = run_raw(mode, img_path)
                prec_raw = calculate_accuracy(raw_text, ground_truth)

                # Post-procesamiento (spellcheck y orden lectura)
                text_post, _ = post_process(raw_text, None)
                prec_post = calculate_accuracy(text_post, ground_truth)

                # Delta de precisión
                delta = prec_post - prec_raw

                # Escribe fila
                writer.writerow([
                    fname,
                    mode,
                    ground_truth,
                    raw_text,
                    f"{prec_raw:.2f}",
                    text_post,
                    f"{prec_post:.2f}",
                    f"{delta:.2f}"
                ])

                print(
                    f"{fname} | {mode} | prec_raw={prec_raw:.2f}% | "
                    f"prec_post={prec_post:.2f}% | delta={delta:.2f}%"
                )

    print(f"\n✅ Benchmark postprocess guardado en {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
