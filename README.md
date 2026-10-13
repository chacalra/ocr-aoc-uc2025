# OCR Evaluation Pipeline

A reproducible Python pipeline for comparing **PaddleOCR** and **Tesseract** on image text recognition tasks with ground-truth evaluation.

The project was developed as academic work in computer vision and OCR. It focuses on three practical questions: how preprocessing affects recognition quality, how bounding-box extraction changes the output, and how simple post-processing influences accuracy.

## What it does

- Runs OCR with **PaddleOCR (PP-OCRv4)** or **Tesseract**.
- Applies OpenCV preprocessing for the Tesseract path.
- Extracts word-level or character-level bounding boxes.
- Reconstructs reading order from detected boxes.
- Applies lightweight spell correction.
- Compares predictions against paired ground-truth text.
- Includes benchmarking scripts and committed benchmark results.

## Pipeline

```text
Input image
    │
    ├── PaddleOCR ───────────────┐
    │                            │
    └── OpenCV preprocessing     │
             │                   │
             └── Tesseract ──────┤
                                  ↓
                         reading-order recovery
                                  ↓
                           post-processing
                                  ↓
                      ground-truth comparison
                                  ↓
                    accuracy + optional boxes
```

## Quick start

Python 3.10+ is recommended. Tesseract must also be installed on the system for the Tesseract modes.

```bash
git clone https://github.com/chacalra/ocr-aoc-uc2025.git
cd ocr-aoc-uc2025
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The default dataset directory is `./Imagenes`. Each image should have a matching `.txt` file containing the expected text:

```text
Imagenes/
├── sample.png
├── sample.txt
├── sample2.jpg
└── sample2.txt
```

## Usage

```bash
python OCR.py MODE [--data-dir PATH]
```

Available modes:

| Mode | Engine | Boxes |
| --- | --- | --- |
| `paddle` | PaddleOCR | No |
| `paddle_boxes` | PaddleOCR | Word |
| `tesseract` | Tesseract | No |
| `tesseract_boxes` | Tesseract | Word |
| `tesseract_chars` | Tesseract | Character |

Examples:

```bash
python OCR.py paddle
python OCR.py paddle_boxes --data-dir ./Imagenes
python OCR.py tesseract --data-dir ./Imagenes
python OCR.py tesseract_chars --data-dir ./Imagenes
```

## Tesseract preprocessing

The Tesseract path uses an image-processing pipeline built with OpenCV:

1. Bicubic upscaling.
2. Grayscale conversion.
3. CLAHE local contrast enhancement.
4. Bilateral filtering.
5. Otsu thresholding.
6. Morphological closing.

The repository also contains an experimental image-specific branch of preprocessing used during evaluation. It is intentionally kept because this project documents an academic exploration rather than presenting the heuristic as a production-general solution.

## Evaluation

Predicted text is compared with the paired ground truth. The current script reports similarity as a percentage using Python's `difflib.SequenceMatcher`.

The committed benchmark results are available at:

- [benchmarks/benchmark_results.csv](benchmarks/benchmark_results.csv)
- [Benchmark documentation](benchmarks/README_BENCHMARKS.md)

For example, the committed results include a sample where PaddleOCR reaches **100%** similarity and Tesseract reaches **98.90%** in raw-text mode. These figures belong to the included evaluation set; they should not be interpreted as general OCR accuracy claims.

## Repository structure

```text
OCR.py                         Main OCR and evaluation script
Imagenes/                      Evaluation images + ground truth
benchmarks/                    Benchmark scripts and results
Resultados/                    Generated/experimental outputs
detect_east3.py                EAST text-detection experiment
requirements.txt               Python dependencies
```

## Scope and limitations

- This is an evaluation/research repository, not a production OCR service.
- The similarity metric is useful for comparative experiments but is not a substitute for CER/WER in a rigorous OCR benchmark.
- The Tesseract preprocessing contains a heuristic tuned for one evaluation image.
- GUI box visualization requires a desktop OpenCV environment.
- OCR quality depends strongly on layout, resolution, typography, and image quality.

## Tech stack

Python · OpenCV · PaddleOCR · Tesseract · NumPy · Pillow · pyspellchecker

## License

MIT
