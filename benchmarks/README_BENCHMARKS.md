# 📊 OCR Benchmark Suite

This repository includes benchmarking scripts to evaluate the performance and accuracy of OCR engines (**PaddleOCR** and **Tesseract**) with and without post-processing.

---

## 📁 Data Setup

Your data folder (default: `./Imagenes`) must contain `.png`, `.jpg`, or `.jpeg` images and corresponding `.txt` ground-truth files.

```
/Imagenes
├── image1.png
├── image1.txt
├── image2.jpg
├── image2.txt
```

---

## 🧪 Benchmarks

### 1. `benchmark_postprocess.py`

Measures the accuracy **before and after** post-processing (spellcheck + line ordering).

#### Output CSV: `benchmark_postprocess.csv`

**Columns:**

- `imagen`: image filename
- `modo`: OCR engine used (`paddle` or `tesseract`)
- `ground-truth`: reference text
- `texto_bruto`: raw OCR result
- `prec_raw`: accuracy before post-processing
- `texto_post`: processed OCR result
- `prec_post`: accuracy after post-processing
- `delta_prec`: difference in accuracy

#### Run it:

```bash
python benchmark_postprocess.py
```

---

### 2. `benchmark_results.py`

Compares OCR modes including bounding box extraction and character-level detection.

#### Output CSV: `benchmark_results.csv`

**Modes Evaluated:**

- `paddle`: PaddleOCR, raw text only
- `paddle_boxes`: PaddleOCR, text + word boxes
- `tesseract`: Tesseract OCR, raw
- `tesseract_boxes`: Tesseract with bounding boxes via `image_to_data`
- `tesseract_chars`: Tesseract character-level OCR

**Columns:**

- `imagen`: image filename
- `modo`: OCR mode used
- `time_ms`: execution time in milliseconds
- `accuracy`: accuracy compared to ground-truth
- `has_boxes`: whether bounding boxes were extracted
- `coords`: JSON with bounding box coordinates

#### Run it:

```bash
python benchmark_results.py
```

---

## 📦 Requirements

Make sure you have all dependencies from `requirements.txt` installed and Tesseract available on your system path.

```bash
pip install -r requirements.txt
```

---

## 📈 Example Output

```
image1.png            | paddle_boxes     | 120.4ms | 94.12% | boxes: True
image1.png            | tesseract        | 103.9ms | 89.45% | boxes: False
image1.png            | tesseract_chars  | 140.2ms | 91.33% | boxes: True
```

---

## ✍️ Author

This benchmarking suite was developed to evaluate OCR performance for structured documents and comic-style text.

---
