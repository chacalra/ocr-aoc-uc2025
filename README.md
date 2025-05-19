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
