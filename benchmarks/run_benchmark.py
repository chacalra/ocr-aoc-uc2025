#!/usr/bin/env python3
import os
import time
import csv
import json
import cv2
from PIL import Image
import numpy as np

from OCR import (
    preprocess_image,
    ocr_with_paddleocr,
    ocr_with_tesseract_data,
    ocr_with_tesseract_chars,
    ocr_with_tesseract,
    calculate_accuracy,
)

# Carpeta con imágenes de prueba y sus .txt de ground truth
DATA_DIR = "/Users/chacalra/Universidaa/9no/SE2/Code/Imagenes"
OUTPUT_CSV = "benchmark_results.csv"

# Modos a testear
MODES = [
    "paddle",
    "paddle_boxes",
    "tesseract",
    "tesseract_boxes",
    "tesseract_chars",
]

def run_mode(mode, image_path):
    """
    Ejecuta el OCR en el modo dado, devuelve (texto, boxes).
    boxes es None o lista de tuplas (coords, texto, confianza).
    """
    if mode == "paddle":
        text = ocr_with_paddleocr(image_path, return_boxes=False)
        boxes = None
    elif mode == "paddle_boxes":
        texts, boxes = ocr_with_paddleocr(image_path, return_boxes=True)
        text = "\n".join(texts)
    else:
        pre = preprocess_image(image_path)
        if mode == "tesseract":
            text = ocr_with_tesseract(pre)
            boxes = None
        elif mode == "tesseract_boxes":
            texts, boxes = ocr_with_tesseract_data(pre, return_boxes=True)
            text = " ".join(texts)
        elif mode == "tesseract_chars":
            text, boxes = ocr_with_tesseract_chars(pre)
        else:
            raise ValueError(f"Modo desconocido: {mode}")
    return text.strip(), boxes


def main():
    # Prepara CSV
    with open(OUTPUT_CSV, "w", newline='', encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        # Agregamos columna 'coords' para guardar las coordenadas
        writer.writerow(["imagen", "modo", "time_ms", "accuracy", "has_boxes", "coords"])

        for fname in sorted(os.listdir(DATA_DIR)):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue

            img_path = os.path.join(DATA_DIR, fname)
            gt_path = os.path.splitext(img_path)[0] + ".txt"
            if not os.path.exists(gt_path):
                print(f"⚠️  No ground-truth para {fname}, saltando")
                continue

            with open(gt_path, "r", encoding="utf-8") as f:
                ground_truth = f.read().strip()

            for mode in MODES:
                t0 = time.time()
                text, boxes = run_mode(mode, img_path)
                elapsed = (time.time() - t0) * 1000  # ms
                acc = calculate_accuracy(text, ground_truth)
                has_boxes = bool(boxes)
                # Extraemos solo las coordenadas de cada box: [[x1,y1],...]
                if boxes:
                    coords = [b[0] for b in boxes]  # b[0] es la lista de 4 puntos
                else:
                    coords = []

                # Serializamos coords a JSON para guardarlo en CSV
                coords_json = json.dumps(coords)
                writer.writerow([fname, mode, f"{elapsed:.1f}", f"{acc:.2f}", has_boxes, coords_json])
                print(f"{fname:20s} | {mode:15s} | {elapsed:6.1f}ms | {acc:6.2f}% | boxes: {has_boxes}")

    print(f"\n✅ Benchmark guardado en {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
