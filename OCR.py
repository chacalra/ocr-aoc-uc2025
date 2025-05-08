import sys
import os
import cv2
import easyocr
import pytesseract
from PIL import Image
import numpy as np
import difflib
from paddleocr import PaddleOCR



# Cargar PaddleOCR (una sola vez)
paddle_ocr_model = PaddleOCR(
    use_angle_cls=True, #correcion de texto
    lang='en', # idioma ingles
    det_model_dir=None,  # dejar None para que use default
    rec_model_dir=None,  # dejar None por ahora (vamos a descargar el server model)
    ocr_version='PP-OCRv4',  # que use versión v4
    use_space_char=True  # para mejorar textos con espacios
)

def draw_boxes(image_path, boxes):
    image = cv2.imread(image_path)
    for box, text, _ in boxes:
        pts = np.array(box).astype(int)
        cv2.polylines(image, [pts], isClosed=True, color=(0, 255, 0), thickness=2)
        cv2.putText(image, text, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    cv2.imshow("Detecciones PaddleOCR", image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()

from pytesseract import Output

def ocr_with_tesseract_data(image_array, return_boxes=False):
    """
    Usa image_to_data para obtener texto y, opcionalmente, boxes de cada palabra/línea.
    """
    pil_img = Image.fromarray(cv2.bitwise_not(image_array))
    data = pytesseract.image_to_data(pil_img, output_type=Output.DICT)
    texts = []
    boxes = []
    n = len(data["text"])
    for i in range(n):
        txt = data["text"][i].strip()
        conf = float(data["conf"][i])
        if not txt:
            continue
        texts.append(txt)
        if return_boxes:
            # coords x, y, w, h
            x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
            box = [[x, y], [x+w, y], [x+w, y+h], [x, y+h]]
            boxes.append((box, txt, conf/100))
    if return_boxes:
        return texts, boxes
    return " ".join(texts)


def ocr_with_tesseract_chars(image_array):
    """
    Usa image_to_boxes para obtener bounding boxes de cada carácter.
    Atención: Tesseract da coordenadas con origen en la esquina inferior izquierda.
    """
    pil_img = Image.fromarray(cv2.bitwise_not(image_array))
    h = image_array.shape[0]
    char_boxes = []
    for line in pytesseract.image_to_boxes(pil_img).splitlines():
        char, x1, y1, x2, y2, _ = line.split()
        x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
        # convertir a coordenadas OpenCV (origen superior izquierdo)
        y1_op = h - y1
        y2_op = h - y2
        box = [[x1, y2_op], [x2, y2_op], [x2, y1_op], [x1, y1_op]]
        char_boxes.append((box, char, None))
    text = pytesseract.image_to_string(pil_img)
    return text.strip(), char_boxes

def ocr_with_paddleocr(path, return_boxes=False):
    result = paddle_ocr_model.ocr(path, cls=True)
    texts = []
    boxes = []

    for line in result[0]:
        box, (text, conf) = line
        texts.append(text)
        if return_boxes:
            boxes.append((box, text, conf))

    if return_boxes:
        return texts, boxes
    return "\n".join(texts)



def upscale_image(image, scale=4):  # ← antes era 3
    height = int(image.shape[0] * scale)
    width = int(image.shape[1] * scale)
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_CUBIC)


def preprocess_image(path):
    import os

    filename = os.path.basename(path)
    image = cv2.imread(path)

    # Escalado general (por defecto x2)
    image = upscale_image(image, scale=3)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)

    print(f"\n🖼️  Preprocesando: {filename}")
    print(f"🔎 Brillo promedio: {avg_brightness:.2f}")

    # Mejora adaptativa por nombre
    if filename == "Imagen 4.png":
        print("🎯 Tratamiento especial para Imagen 4")

        # ↑ Mejorar contraste local
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
        gray = clahe.apply(gray)

        # ↑ Aumentar los bordes
        kernel_sharp = np.array([[0, -1, 0],
                                 [-1, 5, -1],
                                 [0, -1, 0]])
        sharpened = cv2.filter2D(gray, -1, kernel_sharp)

        # ↓ Reducir ruido pero conservar bordes
        filtered = cv2.bilateralFilter(sharpened, d=9, sigmaColor=75, sigmaSpace=75)

        # Umbralización (Otsu)
        _, binary = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Limpieza con morfología
        kernel = np.ones((2, 2), np.uint8)
        processed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

        return processed

    else:
        print("🧠 Fondo claro detectado → modo: estándar")
        # Mejora de contraste suave
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

        blur = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)

        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        kernel = np.ones((2, 2), np.uint8)
        clean = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        return clean


def ocr_with_tesseract(image_array):
    pil_img = Image.fromarray(cv2.bitwise_not(image_array))
    return pytesseract.image_to_string(pil_img)


def ocr_with_easyocr(path):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(path, detail=0)
    return "\n".join(result)


def calculate_accuracy(predicted_text, ground_truth_text):
    seq = difflib.SequenceMatcher(None, predicted_text, ground_truth_text)
    return seq.ratio() * 100


def show_differences(predicted_text, ground_truth_text):
    def make_visible(s):
        return s.replace(" ", "[space]").replace("\n", "[newline]")

    seq = difflib.SequenceMatcher(None, ground_truth_text, predicted_text)
    for tag, i1, i2, j1, j2 in seq.get_opcodes():
        if tag != 'equal':
            gt_part = make_visible(ground_truth_text[i1:i2])
            pred_part = make_visible(predicted_text[j1:j2])
            print(f"{tag.upper()}: Ground Truth [{gt_part}] -> Predicted [{pred_part}]")


# ----------------------------
# PostProcess
# ----------------------------

from spellchecker import SpellChecker  # pip install pyspellchecker
import re

# ————— Inicialización global del corrector ortográfico —————
_spell = SpellChecker(language='en')


def order_reading_lines(boxes, texts, line_tol=10):
    """
    Agrupa y ordena las palabras por orden de lectura.

    boxes: [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] × N
    texts: ["Hello", "world", ...] × N

    Devuelve una lista de líneas, cada línea es lista de tuplas (texto, box):
      [
        [("A", boxA), ("VERY", boxVERY), ...],   # primera línea
        [("BROWN", boxBROWN), ...],              # segunda línea
        ...
      ]
    """
    # 1) calcular centro de cada box
    items = []
    for b, t in zip(boxes, texts):
        ys = [pt[1] for pt in b]
        xs = [pt[0] for pt in b]
        items.append({
            "text": t,
            "box": b,
            "yc": sum(ys) / len(ys),
            "xc": sum(xs) / len(xs)
        })

    # 2) agrupar en líneas por proximidad vertical
    items.sort(key=lambda it: it["yc"])
    lines = []
    current = [items[0]]
    for it in items[1:]:
        if abs(it["yc"] - current[-1]["yc"]) <= line_tol:
            current.append(it)
        else:
            lines.append(current)
            current = [it]
    lines.append(current)

    # 3) ordenar cada línea por coordenada x
    ordered_lines = []
    for line in lines:
        sorted_line = sorted(line, key=lambda it: it["xc"])
        ordered_lines.append([(it["text"], it["box"]) for it in sorted_line])

    return ordered_lines


def simple_spellcheck(text):
    """
    Corrige cada 'token' al diccionario de pyspellchecker,
    preservando puntuación y mayúsculas.
    """
    def _correct(tok):
        # solo corregir si son letras o apóstrofo
        if re.fullmatch(r"[A-Za-z']+", tok):
            low = tok.lower()
            if low not in _spell:
                corr = _spell.correction(low) or low
                return corr
            return low
        # todo lo demás (espacios, puntuación) queda igual
        return tok

    # separa palabras y demás tokens
    tokens = re.findall(r"[A-Za-z']+|[^A-Za-z']+", text)
    corrected = [_correct(tok) for tok in tokens]
    joined = "".join(corrected)
    # colapsar múltiples espacios en uno
    joined = re.sub(r"[ \t]{2,}", " ", joined)
    return joined.upper().strip()


def post_process(text, boxes):
    """
    1) Si hay 'boxes', reordena según lectura y reconstruye líneas.
    2) Si no, toma 'text' tal cual (con sus saltos originales).
    3) Aplica spell-check línea a línea.
    Devuelve (text_proc, boxes_proc).
    """
    if boxes:
        # desempaca
        raw_boxes = [b for b, _, _ in boxes]
        raw_texts = [t for _, t, _ in boxes]

        # obtiene líneas ordenadas
        lines = order_reading_lines(raw_boxes, raw_texts)

        # reconstruye texto multiline
        reconstructed = []
        boxes_proc = []
        for line in lines:
            words = [w for w, _ in line]
            reconstructed.append(" ".join(words))
            for w, box in line:
                boxes_proc.append((box, w, None))

        text_proc = "\n".join(reconstructed)

    else:
        # sin boxes: preserva salto de línea original
        text_proc = text
        boxes_proc = None

    # 3) corrección ortográfica línea a línea
    final_lines = [ simple_spellcheck(ln) for ln in text_proc.splitlines() ]
    text_proc = "\n".join(final_lines)

    return text_proc, boxes_proc




# === USO ===
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ejecuta OCR + post-procesamiento sobre un directorio de imágenes"
    )
    parser.add_argument("mode", choices=[
        "paddle",           # solo texto
        "paddle_boxes",     # texto + cajas palabra
        "tesseract",        # solo texto (con preprocesamiento)
        "tesseract_boxes",  # texto + cajas palabra (con preprocesamiento)
        "tesseract_chars"   # texto + cajas carácter (con preprocesamiento)
    ], help="Modo OCR a utilizar")
    parser.add_argument(
        "--data-dir",
        default="/Users/chacalra/Universidaa/9no/SE2/Code/Imagenes",
        help="Directorio con imágenes y sus .txt de ground truth"
    )
    args = parser.parse_args()
    mode = args.mode
    folder_path = args.data_dir

    print(f"\n📂 Procesando OCR en modo [{mode.upper()}] sobre '{folder_path}'\n")

    # inicializo lista para la precisión global
    all_accuracies = []

    for filename in sorted(os.listdir(folder_path)):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(folder_path, filename)
        gt_path = os.path.splitext(image_path)[0] + ".txt"
        if not os.path.exists(gt_path):
            print(f"⚠️  No existe ground-truth para {filename}, se salta.")
            continue

        # Leer ground-truth (solo para mostrar después la precisión)
        with open(gt_path, "r", encoding="utf-8") as f:
            ground_truth = f.read().strip()

        print(f"\n🖼️  Procesando imagen: {filename}")

        # --- Paso 1: OCR puro ---
        if mode == "paddle":
            text, boxes = ocr_with_paddleocr(image_path, return_boxes=False), None

        elif mode == "paddle_boxes":
            texts, boxes = ocr_with_paddleocr(image_path, return_boxes=True)
            text = "\n".join(texts)
            draw_boxes(image_path, boxes)

        else:
            pre = preprocess_image(image_path)
            if mode == "tesseract":
                text, boxes = ocr_with_tesseract(pre), None

            elif mode == "tesseract_boxes":
                texts, boxes = ocr_with_tesseract_data(pre, return_boxes=True)
                text = " ".join(texts)
                draw_boxes(image_path, boxes)

            elif mode == "tesseract_chars":
                text, boxes = ocr_with_tesseract_chars(pre)
                draw_boxes(image_path, boxes)

            else:
                raise ValueError(f"Modo desconocido: {mode}")

        text = text.strip()

        # --- Paso 2: Post-procesamiento ---
        text_proc, boxes_proc = post_process(text, boxes)

        # --- Salida ---
        print("📄 Texto procesado:\n" + text_proc)

        # Si había cajas, muéstralas ya ordenadas
        if boxes_proc:
            print("\n🗺️  Coordenadas (post-ordenadas):")
            for i, (box, t, _) in enumerate(boxes_proc, 1):
                print(f"   {i}. '{t}' → {box}")

        # Opcional: comparar con ground-truth
        acc = calculate_accuracy(text_proc, ground_truth)
        all_accuracies.append(acc)
        print(f"\n🎯 Precisión vs. ground-truth: {acc:.2f}%")
        print("-" * 60)

        # ---- Debug  ------
        # print("🔍 Diferencias detectadas:")
        # show_differences(text, ground_truth)

    # ----------------------------
    # Precisión promedio global
    # ----------------------------
    if all_accuracies:
        avg = sum(all_accuracies) / len(all_accuracies)
        print(f"\n✅ Precisión promedio global: {avg:.2f}%")
    else:
        print("\n⚠️ No se pudo calcular precisión global (ninguna imagen válida).")





