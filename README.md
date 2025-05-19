# Install

# OCR Pipeline: PaddleOCR / Tesseract + Post-Processing + Metrics

This script (`OCR.py`) performs Optical Character Recognition (OCR) using **PaddleOCR** or **Tesseract**, including image preprocessing, spell correction, line ordering, bounding box visualization, and accuracy evaluation against ground-truth files.

## 📦 Requirements

Install all required dependencies using the `requirements.txt` file:

```bash
pip install -r requirements.txt
```


## 📂 Expected Data Directory Structure

Your data folder (default: `./Imagenes`) should contain `.png`, `.jpg`, or `.jpeg` images **along with** `.txt` files with the same name (without extension) that represent the expected text (*ground truth*).  
Example:

```
/Imagenes
├── example1.png
├── example1.txt
├── image2.jpg
├── image2.txt
```

## 🚀 Usage

```bash
python OCR.py [paddle|tesseract] [--data-dir PATH] [--show-boxes]
```

### Parameters

- `paddle` | `tesseract`: Select the OCR engine to use.
- `--data-dir`: Path to the folder containing images and ground-truth text files (default: `./Imagenes`).
- `--show-boxes`: Display detected text boxes for 3.5 seconds per image.

### Examples

Run OCR using PaddleOCR:

```bash
python OCR.py paddle --data-dir ./Imagenes --show-boxes
```

Run OCR using Tesseract:

```bash
python OCR.py tesseract
```

## ⚙️ Features

### 🔍 Preprocessing
- Image upscaling.
- Grayscale conversion.
- CLAHE (adaptive histogram equalization).
- Bilateral filtering and sharpening.
- Binary thresholding + morphological closing.

### 🧠 OCR Engines
- **PaddleOCR**: Uses `PaddleOCR(use_angle_cls=True, lang='en', ocr_version='PP-OCRv4')`.
- **Tesseract**: Applied over the preprocessed image, detects words and characters.

### 🪄 Post-processing
- Word ordering based on vertical clustering.
- English spell-check correction (`pyspellchecker`).
- Converts final output to uppercase clean text.

### 📐 Visualization
- Draws word and/or character bounding boxes on images.

### 🎯 Metrics
- Accuracy is computed using `difflib.SequenceMatcher` between final output and ground-truth text.

## 📊 Output

- For each image:
  - Final corrected text.
  - Word and/or character boxes (if `--show-boxes` is used).
  - Individual accuracy.
- At the end, displays the **average global accuracy** across all images.

---

## 📝 Example Output

```
📂  Mode [PADDLE] in «./Imagenes»

🖼️  example1.png
📄  Final text:
THE QUICK BROWN FOX JUMPS OVER THE LAZY DOG

🎯  Accuracy: 95.83%
------------------------------------------------------------

✅  Average accuracy: 93.12%
```

---

## 🤝 Author

This code was developed as part of an OCR evaluation pipeline.  
For educational and academic use. You can adapt or extend it with other engines, layout analysis, or performance evaluations.

---


# Running benchmarks


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


